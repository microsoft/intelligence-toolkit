"""
Agentic exploration strategy for Schemify — hybrid phased approach.

Combines the agent's intelligence with the combinatorial approach's discipline.

Three phases with structurally enforced budget splits:
  Phase 1 — Broad Discovery (60%): Agent picks from generated search angles
    + its own creative queries. All parallel. Tracks productive/zero-yield angles.
  Phase 2 — Targeted Discovery (20%): Agent sees which Phase 1 angles were
    productive and generates intersectional queries (like pairs from productive
    singles). Still parallel.
  Phase 3 — Completion (20%): Agent picks entities closest to complete. All
    completions run in PARALLEL with a semaphore.

Uses gpt-5.2 with reasoning for the planning/decision calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Optional

from .models import (
    Record, RecordSet, SchemaAttribute, AttributeValue,
    SchemifyConfig, BudgetExceededError,
)
from .llm import LLMClient
from .extraction import ExtractionEngine
from .resolution import ResolutionEngine

logger = logging.getLogger("schemify.agentic")


# ---------------------------------------------------------------------------
# Phase-specific prompts — replace the single monolithic prompt
# ---------------------------------------------------------------------------

# Shared instructions for normalizations and schema changes (all phases)
_SHARED_TOOLS_INSTRUCTIONS = """\
### `normalizations` — Merge/clean attribute values (0-10 per iteration)
Fix inconsistencies as you see them — don't let them accumulate:
- Merge synonyms: "NGO" + "Non-profit" → "Non-profit"
- Standardize regions: "UK" → "United Kingdom", "Thailand" → "Southeast Asia"
- Fix formatting: "machine learning" → "Machine Learning"
- Only normalize values for attributes listed in the schema

### `schema_changes` — Curate the schema (0-5 per iteration)
You control which attributes are core. Each change has a `kind`:

- **`demote`**: Remove an attribute from the core schema. Provide `attribute` name.
- **`promote`**: Add a frequently-seen additional attribute to core. Provide `attribute` name.
- **`rename`**: Fix a poorly-named attribute. Provide `attribute` (old) and `new_name`.
- **`decompose`**: Break a compound value into atomic pieces. Provide `source_attribute`,
  `compound_value`, and `replacements` (list of dicts with "attribute" and "value").

Schema quality guidelines:
- Each attribute should measure exactly ONE dimension
- If an attribute has <20% fill rate after 2+ iterations, consider demoting it
- Core schema should stay small (5-8 attributes)
"""

# ── Phase 1: Broad Discovery ──

PHASE1_DISCOVERY_PROMPT = """\
You are a research strategist directing a web-search-based entity extraction project.
Your job: examine the current state and issue broad discovery queries to find as many
entities as possible.

**Today's date:** {current_date}. Treat your model knowledge as potentially stale
and prefer queries that surface up-to-date sources (recent years, current versions,
live websites). Do not assume entities you remember still exist or still have the
same attributes.

## Project

**Category:** {category}
**Guidance:** {guidance}

## Schema Attributes

{schema_summary}

## Current State ({entity_count} entities discovered)

{state_summary}

## Search Angles (systematic coverage — suggested queries)

These are auto-generated search angles from your schema's closed-set attribute values.
Pick the most promising ones (you can use them verbatim or adapt them), AND add your
own creative queries to cover niches the angles miss.

{search_angles}

## Query History (last {history_window} of {total_queries} queries)

{query_history}

## Zero-Yield Queries (avoid repeating these)

{zero_yield_summary}

## Budget

- Queries used: {queries_used} / {max_queries} (Phase 1 discovery)
- Estimated cost: ${cost:.2f} / ${max_budget:.2f}

## Instructions — Phase 1: BROAD DISCOVERY

Your PRIMARY goal is to **discover as many entities as possible**.
Do NOT prioritize completions — that comes in Phase 3.

### `queries` — Discovery searches (5-15 per iteration)
Each query becomes a grounded web search. Write queries that DESCRIBE the
category/niche you want to find — do NOT name specific products you know.

- BAD: "PostgreSQL open-source database" (names a known product)
- BAD: "Find databases" (too vague)
- GOOD: "open-source columnar analytics databases used by retailers"
- GOOD: "embedded time-series databases for IoT edge devices"
- GOOD: "vector databases optimized for similarity search in finance"

Strategy:
1. Start with the search angles — pick promising ones you haven't tried yet
2. Add creative queries the angles miss (geographic niches, new technology types)
3. Aim for BREADTH — cover different technology types, regions, and stakeholders
4. Avoid queries similar to zero-yield ones listed above

""" + _SHARED_TOOLS_INSTRUCTIONS + """
### `completions` — NOT available in Phase 1. Leave empty.

### `stop` — Whether to continue
Set to true ONLY when recent queries are mostly yielding duplicates
OR budget is nearly exhausted.

**Unverified seeds**: Some entities may have been seeded from model knowledge
(marked "Unverified"). Use category-descriptive queries to find tools the seeds
missed — don't just confirm what's seeded.
"""

# ── Phase 2: Targeted Discovery ──

PHASE2_TARGETED_PROMPT = """\
You are a research strategist directing a web-search-based entity extraction project.
Phase 1 (broad discovery) is complete. Now you're generating targeted, intersectional
queries to fill coverage gaps.

**Today's date:** {current_date}. Prefer queries that surface current sources;
your model knowledge may be stale.

## Project

**Category:** {category}
**Guidance:** {guidance}

## Schema Attributes

{schema_summary}

## Current State ({entity_count} entities discovered)

{state_summary}

## Productive Search Angles (from Phase 1)

These angles found the most entities — combine them for intersectional queries:

{productive_angles}

## Zero-Yield Search Angles (avoid these)

{zero_yield_summary}

## Query History (last {history_window} of {total_queries} queries)

{query_history}

## Budget

- Queries used: {queries_used} / {max_queries} (Phase 2 targeted discovery)
- Estimated cost: ${cost:.2f} / ${max_budget:.2f}

## Instructions — Phase 2: TARGETED DISCOVERY

Phase 1 found {entity_count} entities. Now find entities that fell through the cracks.

### `queries` — Targeted searches (5-10 per iteration)
Write intersectional queries that **combine** two productive angles from Phase 1.
Also target under-represented niches visible in the value distributions.

Strategy:
1. Combine productive angles: if "mobile apps Southeast Asia" and "AI law enforcement"
   both found entities, try "AI-powered mobile tools for law enforcement in Southeast Asia"
2. Look for gaps in value distributions — regions/types/platforms with few entities
3. Target specific under-represented niches
4. Do NOT repeat queries similar to zero-yield ones

""" + _SHARED_TOOLS_INSTRUCTIONS + """
### `completions` — Light completions only (0-5 per iteration)
You may request a few completions for entities that are VERY close to complete
(only 1-2 missing attributes with 0 prior attempts). This is secondary to discovery.

### `stop` — Whether to continue
Set to true when recent queries are mostly yielding duplicates.
"""

# ── Phase 3: Completion ──

PHASE3_COMPLETION_PROMPT = """\
You are a research strategist directing a web-search-based entity extraction project.
Phases 1 & 2 (discovery) are complete. Now you're filling gaps on existing entities.

**Today's date:** {current_date}. Prefer queries that surface current sources;
your model knowledge may be stale.

## Project

**Category:** {category}
**Guidance:** {guidance}

## Schema Attributes

{schema_summary}

## Current State ({entity_count} entities discovered)

{state_summary}

## Budget

- Queries used: {queries_used} / {max_queries} (Phase 3 completion)
- Estimated cost: ${cost:.2f} / ${max_budget:.2f}

## Instructions — Phase 3: COMPLETION

Focus on building the **most complete records** possible.

### `completions` — Fill gaps on specific entities (10-30 per iteration)
Name entities that are missing critical attributes and should be searched specifically.

**Attempt tracking**: The state summary shows completion attempts per entity+attribute.
Do NOT request completions for attributes with 2+ prior attempts.

Strategy:
1. Prefer entities closest to complete (7/9 attrs → only needs 2 searches)
2. Focus on attributes with 0 prior attempts
3. Skip entities where all missing attrs have 2+ attempts (exhausted)
4. Batch as many completions as your budget allows — they run in PARALLEL

""" + _SHARED_TOOLS_INSTRUCTIONS + """
### `queries` — Discovery searches (0-3 per iteration)
You may issue a few discovery queries if you notice specific niches still missing,
but completions should be your primary tool.

### `stop` — Whether to continue
Set to true when most remaining gaps have 2+ attempts OR budget is exhausted.
"""

AGENT_PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step analysis of current gaps and strategy"
                },
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Specific, targeted web search query"
                            },
                            "goal": {
                                "type": "string",
                                "description": "What gap this fills (geographic, attribute, stakeholder, etc.)"
                            }
                        },
                        "required": ["search_query", "goal"],
                        "additionalProperties": False
                    },
                    "description": "5-15 targeted discovery queries"
                },
                "normalizations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "attribute": {
                                "type": "string",
                                "description": "Schema attribute name"
                            },
                            "merge_values": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Values to merge (2+)"
                            },
                            "canonical": {
                                "type": "string",
                                "description": "The canonical value to keep"
                            }
                        },
                        "required": ["attribute", "merge_values", "canonical"],
                        "additionalProperties": False
                    },
                    "description": "Value normalizations to apply now"
                },
                "completions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "Entity label to complete"
                            },
                            "missing_attributes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Attributes to search for"
                            }
                        },
                        "required": ["entity", "missing_attributes"],
                        "additionalProperties": False
                    },
                    "description": "Specific entities to fill gaps on (up to 30)"
                },
                "schema_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["demote", "promote", "rename", "decompose"],
                                "description": "Type of schema change"
                            },
                            "attribute": {
                                "type": "string",
                                "description": "Attribute name (for demote/promote/rename) or source attribute (for decompose)"
                            },
                            "new_name": {
                                "type": "string",
                                "description": "New attribute name (for rename). Empty string if not applicable."
                            },
                            "compound_value": {
                                "type": "string",
                                "description": "The compound value to decompose (for decompose). Empty string if not applicable."
                            },
                            "replacements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "attribute": {
                                            "type": "string",
                                            "description": "Target attribute for this piece"
                                        },
                                        "value": {
                                            "type": "string",
                                            "description": "Atomic value for this piece"
                                        }
                                    },
                                    "required": ["attribute", "value"],
                                    "additionalProperties": False
                                },
                                "description": "Atomic pieces for decompose. Empty array if not applicable."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this schema change is needed"
                            }
                        },
                        "required": ["kind", "attribute", "new_name", "compound_value", "replacements", "reason"],
                        "additionalProperties": False
                    },
                    "description": "Schema curation actions (demote/promote/rename/decompose)"
                },
                "stop": {
                    "type": "boolean",
                    "description": "True if exploration should stop"
                }
            },
            "required": ["reasoning", "queries", "normalizations", "completions", "schema_changes", "stop"],
            "additionalProperties": False
        }
    }
}


# ---------------------------------------------------------------------------
# Seed-from-knowledge prompt — generates candidate entities from LLM memory
# ---------------------------------------------------------------------------

SEED_KNOWLEDGE_PROMPT = """\
You are an expert researcher. Given a category and guidance, list every entity
you can confidently name from your own knowledge. For each entity, fill in as
many schema attributes as you know. These will be treated as UNVERIFIED
candidates — web searches will confirm or correct them later.

## Category
{category}

## Guidance
{guidance}

## Schema Attributes
{schema_summary}

## Instructions
- List **as many real, specific, named entities** as you can (target 20-60).
- Fill attribute values ONLY if you are reasonably confident — leave empty string "" if unsure.
- Use the exact attribute names from the schema.
- Include a `confidence_note` for each entity: "high" if well-known, "medium" if
  you recall the name but are fuzzy on details, "low" if you're uncertain it exists
  or is correctly categorised.
- Do NOT invent entities. Only list things you believe to be real.
- Prefer breadth across different technology types, geographies, and stakeholders.
"""

SEED_KNOWLEDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "seed_entities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Entity name"
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Alternative names"
                            },
                            "attributes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Schema attribute name (must match exactly)"
                                        },
                                        "value": {
                                            "type": "string",
                                            "description": "Attribute value (empty string if unknown)"
                                        }
                                    },
                                    "required": ["name", "value"],
                                    "additionalProperties": False
                                },
                                "description": "Known attribute values for this entity"
                            },
                            "confidence_note": {
                                "type": "string",
                                "description": "high, medium, or low"
                            }
                        },
                        "required": ["label", "aliases", "attributes", "confidence_note"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
    }
}


# ---------------------------------------------------------------------------
# State summary builders
# ---------------------------------------------------------------------------

def build_schema_summary(record_set: RecordSet) -> str:
    """Compact schema description for the agent, including fill rates."""
    lines = []
    total = len(record_set.records) if record_set.records else 0
    for attr in record_set.schema_attributes:
        desc = attr.description or ""
        closed = "closed" if attr.is_closed_set else "open"
        n_vals = len(attr.provisional_values) if attr.provisional_values else 0
        if total > 0:
            filled = sum(
                1 for r in record_set.records
                if r.attributes.get(attr.name) and r.attributes[attr.name].value
            )
            pct = filled * 100 // total
            lines.append(f"- {attr.name} ({closed}, {n_vals} known values, {pct}% filled): {desc}")
        else:
            lines.append(f"- {attr.name} ({closed}, {n_vals} known values): {desc}")
    return "\n".join(lines) if lines else "(no schema defined yet)"


def build_state_summary(record_set: RecordSet, max_entities: int = 50) -> str:
    """
    Build a compact state summary showing:
    - Attribute coverage stats
    - Value distribution for each attribute
    - Sample entities with gaps
    """
    if not record_set.records:
        return "(no entities discovered yet)"

    schema_attrs = [a.name for a in record_set.schema_attributes]
    records = record_set.records
    total = len(records)

    parts = []

    # Per-attribute coverage and value distribution
    parts.append("### Attribute Coverage & Value Distribution\n")
    for attr_name in schema_attrs:
        filled = sum(1 for r in records if r.attributes.get(attr_name) and r.attributes[attr_name].value)
        pct = filled / total * 100 if total else 0

        # Value distribution (top values)
        values: Counter = Counter()
        for r in records:
            av = r.attributes.get(attr_name)
            if av and av.value:
                for v in av.value.split("|"):
                    v = v.strip()
                    if v:
                        values[v] += 1

        top = values.most_common(8)
        dist = ", ".join(f"{v} ({c})" for v, c in top)
        tail = f" + {len(values) - 8} more" if len(values) > 8 else ""
        parts.append(f"**{attr_name}**: {filled}/{total} ({pct:.0f}%) — {dist}{tail}")

    # Entities with worst coverage
    parts.append("\n### Entities with Most Missing Attributes\n")
    scored = []
    for r in records:
        missing = [a for a in schema_attrs if not r.attributes.get(a) or not r.attributes[a].value]
        if missing:
            scored.append((r.label, missing))
    scored.sort(key=lambda x: -len(x[1]))
    for label, missing in scored[:10]:
        parts.append(f"- **{label}**: missing {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

    # Completion attempts summary
    entities_with_attempts = []
    total_exhausted = 0  # attrs with 2+ attempts and still missing
    total_untried = 0    # missing attrs with 0 attempts
    for r in records:
        missing_attrs = [a for a in schema_attrs if not r.attributes.get(a) or not r.attributes[a].value]
        if not missing_attrs:
            continue
        attempted_details = []
        for a in missing_attrs:
            n = r.completion_attempts.get(a, 0)
            if n >= 2:
                total_exhausted += 1
            elif n == 0:
                total_untried += 1
            if n > 0:
                attempted_details.append(f"{a} ({n}x)")
        if attempted_details:
            entities_with_attempts.append((r.label, attempted_details))

    parts.append(f"\n### Completion Attempt Tracking\n")
    parts.append(f"Missing attributes with 0 prior attempts (actionable): {total_untried}")
    parts.append(f"Missing attributes with 2+ prior attempts (exhausted — skip): {total_exhausted}")
    if entities_with_attempts:
        parts.append(f"\nEntities with prior failed attempts:")
        for label, details in entities_with_attempts[:15]:
            parts.append(f"- **{label}**: {', '.join(details)}")
        if len(entities_with_attempts) > 15:
            parts.append(f"  ... +{len(entities_with_attempts) - 15} more")

    # Entities closest to complete (best candidates for completions)
    almost_complete = []
    for r in records:
        filled = sum(1 for a in schema_attrs if r.attributes.get(a) and r.attributes[a].value)
        missing = [a for a in schema_attrs if not r.attributes.get(a) or not r.attributes[a].value]
        # Filter out exhausted attrs (2+ attempts)
        actionable = [a for a in missing if r.completion_attempts.get(a, 0) < 2]
        if actionable and filled > 0:
            almost_complete.append((r.label, filled, len(schema_attrs), actionable))
    almost_complete.sort(key=lambda x: -x[1])  # Most-filled first
    if almost_complete:
        parts.append(f"\n### Entities Closest to Complete (best completion targets)\n")
        for label, filled, total_a, actionable in almost_complete[:15]:
            parts.append(f"- **{label}** ({filled}/{total_a}): needs {', '.join(actionable[:4])}{'...' if len(actionable) > 4 else ''}")

    # Entity sample (all labels for dedup awareness)
    # Flag unverified seeds (no sources on any attribute)
    verified_labels = []
    unverified_labels = []
    for r in records:
        has_sources = any(
            av.sources for av in r.attributes.values() if av
        )
        if has_sources:
            verified_labels.append(r.label)
        else:
            unverified_labels.append(r.label)

    parts.append(f"\n### All Entities ({total} total)\n")
    if unverified_labels:
        shown = unverified_labels[:max_entities]
        tail = f" ... and {len(unverified_labels) - max_entities} more" if len(unverified_labels) > max_entities else ""
        parts.append(f"**Unverified seeds ({len(unverified_labels)}):** {', '.join(shown)}{tail}")
    if verified_labels:
        shown = verified_labels[:max_entities]
        tail = f" ... and {len(verified_labels) - max_entities} more" if len(verified_labels) > max_entities else ""
        parts.append(f"**Verified ({len(verified_labels)}):** {', '.join(shown)}{tail}")

    return "\n".join(parts)


def build_query_history(history: list[dict], window: int = 15) -> str:
    """Format recent query history for the agent."""
    if not history:
        return "(no queries executed yet)"
    recent = history[-window:]
    lines = []
    for h in recent:
        new = h.get("new_entities", 0)
        dups = h.get("duplicates", 0)
        focus = h.get("focus", "")[:80]
        lines.append(f"- [{new} new, {dups} dups] {focus}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search-angle generation (from attribute values, like combinatorial singles)
# ---------------------------------------------------------------------------

def generate_search_angles(record_set: RecordSet, category: str) -> list[dict]:
    """
    Generate systematic search angles from closed-set attribute values.

    This mirrors the combinatorial approach's single-attribute queries:
    for each closed-set attribute with exploration/provisional values,
    generate a natural-language search query combining the category
    with that specific value.

    Returns list of dicts with 'query', 'attribute', 'value', 'used', 'yield'.
    """
    angles: list[dict] = []
    for attr in record_set.schema_attributes:
        values = attr.exploration_values or attr.provisional_values or []
        if not values:
            continue
        for val in values[:15]:  # Cap at 15 values per attribute
            query = f"{category} where {attr.name} is {val}"
            angles.append({
                "query": query,
                "attribute": attr.name,
                "value": val,
                "used": False,
                "yield": 0,
            })
    return angles


def format_search_angles(
    angles: list[dict],
    used_queries: set[str] | None = None,
    max_show: int = 40,
) -> str:
    """Format search angles for the agent prompt, marking used ones."""
    if not angles:
        return "(no search angles generated — schema has no closed-set attributes with values)"

    unused = [a for a in angles if not a.get("used")]
    used = [a for a in angles if a.get("used")]

    lines = []
    if unused:
        lines.append(f"**Available ({len(unused)} unused):**")
        for a in unused[:max_show]:
            lines.append(f"- [{a['attribute']}={a['value']}] {a['query']}")
        if len(unused) > max_show:
            lines.append(f"  ... +{len(unused) - max_show} more")

    if used:
        productive = [a for a in used if a.get("yield", 0) > 0]
        barren = [a for a in used if a.get("yield", 0) == 0]
        if productive:
            lines.append(f"\n**Already used — productive ({len(productive)}):**")
            for a in sorted(productive, key=lambda x: -x.get("yield", 0))[:10]:
                lines.append(f"- [{a['attribute']}={a['value']}] → {a['yield']} new entities")
        if barren:
            lines.append(f"\n**Already used — zero yield ({len(barren)}):** "
                        f"{', '.join(a['value'] for a in barren[:15])}"
                        f"{'...' if len(barren) > 15 else ''}")

    return "\n".join(lines) if lines else "(none)"


def format_productive_angles(angles: list[dict]) -> str:
    """Format productive angles for Phase 2 targeted prompt."""
    productive = [a for a in angles if a.get("yield", 0) > 0]
    if not productive:
        return "(no productive angles from Phase 1)"

    productive.sort(key=lambda x: -x.get("yield", 0))
    lines = []
    for a in productive[:20]:
        lines.append(f"- [{a['attribute']}={a['value']}] → {a['yield']} entities")
    return "\n".join(lines)


def format_zero_yield_summary(
    zero_yield_queries: list[str],
    angles: list[dict] | None = None,
) -> str:
    """Format zero-yield queries and angles for the agent."""
    parts = []
    barren_angles = [a for a in (angles or []) if a.get("used") and a.get("yield", 0) == 0]
    if barren_angles:
        parts.append(f"**Zero-yield angles ({len(barren_angles)}):** "
                    f"{', '.join(a['value'] for a in barren_angles[:20])}")
    if zero_yield_queries:
        parts.append(f"**Zero-yield free-form queries ({len(zero_yield_queries)}):**")
        for q in zero_yield_queries[:10]:
            parts.append(f"- {q[:80]}")
    if not parts:
        return "(none yet)"
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dashboard export
# ---------------------------------------------------------------------------

def export_dashboard_js(record_set: RecordSet, filepath: str) -> int:
    """
    Export a RecordSet as a slim ``dashboard_data.js`` file.

    Strips internal bookkeeping (tier, retrieved_at, title, confidence,
    additional_attributes, alias_counts, etc.) to produce a minified JS
    file suitable for loading from a local filesystem via ``<script src>``.

    Args:
        record_set: The RecordSet to export.
        filepath: Destination path (e.g. ``output/dashboard_data.js``).

    Returns:
        File size in bytes.
    """
    slim_records = []
    for rec in record_set.records:
        r: dict[str, Any] = {"label": rec.label}
        if rec.aliases:
            r["aliases"] = rec.aliases
        attrs: dict[str, Any] = {}
        for aname, aval in rec.attributes.items():
            slim_vals = []
            for sv in aval.values:
                slim_sources = []
                for src in sv.sources:
                    ss: dict[str, str] = {"url": src.url}
                    if src.snippet:
                        ss["snippet"] = src.snippet
                    slim_sources.append(ss)
                slim_vals.append({"value": sv.value, "sources": slim_sources})
            attrs[aname] = {"value": aval.value, "values": slim_vals}
        r["attributes"] = attrs
        slim_records.append(r)

    slim_data = {
        "category": record_set.category,
        "records": slim_records,
        "schema_attributes": [{"name": a.name} for a in record_set.schema_attributes],
    }

    js = "var DASHBOARD_DATA = " + json.dumps(
        slim_data, ensure_ascii=False, separators=(",", ":")
    ) + ";"

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(js)

    size = os.path.getsize(filepath)
    logger.info(f"Exported dashboard JS: {filepath} ({size / 1024:.0f} KB, {len(slim_records)} records)")
    return size


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

class AgenticStrategy:
    """
    Agent-driven exploration strategy.

    Each iteration:
    1. Build state summary from RecordSet
    2. Ask reasoning model to plan next actions
    3. Execute discovery queries in parallel
    4. Apply normalizations
    5. Run targeted completions
    6. Repeat until agent says stop or budget exhausted
    """

    def __init__(
        self,
        config: SchemifyConfig,
        llm: LLMClient,
        extraction: ExtractionEngine,
        resolution: ResolutionEngine,
    ):
        self.config = config
        self.llm = llm
        self.extraction = extraction
        self.resolution = resolution
        self._log_file = None  # set during run()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _emit(self, msg: str, *, bold: bool = False, dim: bool = False):
        """Print to notebook *and* append to log file."""
        # Print to stdout (visible in notebook)
        prefix = ""
        if bold:
            prefix = "\033[1m"   # ANSI bold
        elif dim:
            prefix = "\033[2m"  # ANSI dim
        suffix = "\033[0m" if (bold or dim) else ""
        print(f"{prefix}{msg}{suffix}", flush=True)

        # Append to log file if open
        if self._log_file:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    # ------------------------------------------------------------------
    # Seed from knowledge
    # ------------------------------------------------------------------

    async def _seed_from_knowledge(self, record_set: RecordSet) -> int:
        """
        Ask the LLM to list candidate entities from its training knowledge.
        Returns count of entities added. All are marked as unverified with
        low confidence so the agent prioritises verifying them.
        """
        self._emit("Seeding candidate entities from model knowledge...", bold=True)
        self.llm.set_progress_context("Seed from knowledge")

        # Use medium reasoning for this planning-style call
        original_effort = self.config.reasoning_effort
        self.config.reasoning_effort = "medium"

        try:
            result = await self.llm.structured_completion(
                prompt=SEED_KNOWLEDGE_PROMPT,
                response_format=SEED_KNOWLEDGE_SCHEMA,
                variables={
                    "category": record_set.category,
                    "guidance": record_set.guidance or "",
                    "schema_summary": build_schema_summary(record_set),
                },
            )
        finally:
            self.config.reasoning_effort = original_effort
            self.llm.set_progress_context("")

        entities = result.get("entities", [])
        schema_attr_names = {a.name for a in record_set.schema_attributes}
        added = 0

        for ent in entities:
            label = ent.get("label", "").strip()
            if not label:
                continue

            confidence_note = ent.get("confidence_note", "medium")
            base_conf = {"high": 0.3, "medium": 0.2, "low": 0.1}.get(confidence_note, 0.2)

            rec = Record(label=label)
            rec.aliases = [a.strip() for a in ent.get("aliases", []) if a.strip()]

            for attr in ent.get("attributes", []):
                attr_name = attr.get("name", "")
                val = attr.get("value", "")
                if not val or attr_name not in schema_attr_names:
                    continue
                av = AttributeValue()
                av.add_value(val)  # No source — unverified
                av.confidence = base_conf
                av.verified = False
                rec.set_attribute(attr_name, av, is_schema_attr=True)

            was_new, _ = record_set.add_record(rec, use_fuzzy=True)
            if was_new:
                added += 1

        self._emit(
            f"  Seeded {added} candidate entities from model knowledge "
            f"({len(entities) - added} duplicates filtered)",
            bold=True,
        )
        return added

    async def run(
        self,
        record_set: RecordSet,
        max_queries: int = 100,
        concurrency: int = 5,
        output_dir: str | None = None,
        seed_state: str | None = None,
        seed_records: str | list[dict] | "pd.DataFrame" | None = None,
        phase_split: tuple[float, float, float] = (0.60, 0.20, 0.20),
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[dict]:
        """
        Run the hybrid phased exploration loop.

        Phase 1 — Broad Discovery (phase_split[0] of budget):
          Agent picks from generated search angles + its own creative queries.
          All queries run in parallel. Tracks productive/zero-yield angles.
          No completions.

        Phase 2 — Targeted Discovery (phase_split[1] of budget):
          Agent sees Phase 1 productive angles and generates intersectional
          queries. Light completions allowed (entities nearly complete).

        Phase 3 — Completion (phase_split[2] of budget):
          Agent picks entities to complete.  All completions run in PARALLEL.
          Light discovery for missed niches.

        Args:
            record_set: The RecordSet to populate
            max_queries: Total query budget (all phases)
            concurrency: Max parallel web searches
            output_dir: Where to save snapshots and logs
            seed_state: Path to a Schemify JSON save file to restore
            seed_records: Additional data to merge
            phase_split: Budget fractions for (discovery, targeted, completion)

        Returns:
            List of iteration history dicts
        """
        # Restore save-state first
        if seed_state is not None:
            restored = self._restore_state(record_set, seed_state)
            self._emit(
                f"Restored state: {restored['records']} records, "
                f"{restored['schema_attrs']} schema attributes from {seed_state}",
                bold=True,
            )

        # Layer additional seed records
        if seed_records is not None:
            ingested = await self._ingest_seed_records(record_set, seed_records)
            self._emit(f"Seeded {ingested} additional records from prior data", bold=True)

        # Seed from model knowledge when starting from scratch
        if not record_set.records:
            await self._seed_from_knowledge(record_set)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.llm.set_output_dir(output_dir)
            log_path = os.path.join(output_dir, "agentic_log.txt")
            self._log_file = open(log_path, "a", encoding="utf-8")
        else:
            self._log_file = None

        # Calculate phase budgets
        p1_budget = max(1, int(max_queries * phase_split[0]))
        p2_budget = max(1, int(max_queries * phase_split[1]))
        p3_budget = max(1, max_queries - p1_budget - p2_budget)

        self._emit(f"{'='*60}", bold=True)
        self._emit(f"Hybrid Phased Run: {datetime.now().isoformat()}")
        self._emit(f"Category: {record_set.category}")
        self._emit(f"Max queries: {max_queries}, Concurrency: {concurrency}")
        self._emit(f"Phase budgets: P1={p1_budget} discovery, P2={p2_budget} targeted, P3={p3_budget} completion")
        self._emit(f"Starting entities: {len(record_set.records)}")
        self._emit(f"{'='*60}")

        # Generate search angles from closed-set attribute values
        search_angles = generate_search_angles(record_set, record_set.category)
        self._emit(f"Generated {len(search_angles)} search angles from schema values")

        history: list[dict] = []
        queries_used = 0
        total_queries_used = 0
        consecutive_all_timeout_iters = 0
        zero_yield_queries: list[str] = []  # Free-form queries that found nothing

        def _emit_progress(phase: str, phase_used: int, phase_budget: int) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(
                    f"{phase} ({phase_used}/{phase_budget})",
                    total_queries_used + phase_used,
                    max_queries,
                )
            except Exception:  # noqa: BLE001
                pass

        try:
            # ══════════════════════════════════════════════════════════
            # PHASE 1: Broad Discovery
            # ══════════════════════════════════════════════════════════
            self._emit(f"\n{'═'*60}", bold=True)
            self._emit(f"PHASE 1: BROAD DISCOVERY (budget: {p1_budget} queries)", bold=True)
            self._emit(f"{'═'*60}")

            p1_used = 0
            iteration = 0

            while p1_used < p1_budget:
                iteration += 1
                iter_start = datetime.now()

                self._emit(f"\n{'─'*60}", bold=True)
                self._emit(
                    f"P1 ITERATION {iteration}  │  {p1_used}/{p1_budget} queries  │  "
                    f"${self.llm.total_cost:.2f}  │  {len(record_set.records)} entities",
                    bold=True,
                )
                self._emit(f"{'─'*60}")

                plan = await self._get_plan(
                    record_set=record_set,
                    history=history,
                    queries_used=p1_used,
                    max_queries=p1_budget,
                    phase="discovery",
                    search_angles=search_angles,
                    zero_yield_queries=zero_yield_queries,
                )

                reasoning = plan.get("reasoning", "")
                discovery_queries = plan.get("queries", [])[:15]
                normalizations = plan.get("normalizations", [])
                schema_changes = plan.get("schema_changes", [])[:5]
                should_stop = plan.get("stop", False)

                self._emit(f"\n🧠 Agent reasoning:")
                for line in reasoning.split("\n"):
                    self._emit(f"   {line}")
                self._emit(
                    f"\n→ Plan: {len(discovery_queries)} queries, "
                    f"{len(normalizations)} normalizations, "
                    f"{len(schema_changes)} schema changes"
                    f"{' [STOP signalled]' if should_stop else ''}"
                )

                if should_stop and not discovery_queries:
                    self._emit("\nAgent decided to stop Phase 1.", bold=True)
                    break

                # Apply schema changes & normalizations
                if schema_changes:
                    applied = self._apply_schema_changes(record_set, schema_changes)
                    self._emit(f"  Applied {applied} schema changes")
                if normalizations:
                    applied = self._apply_normalizations(record_set, normalizations)
                    self._emit(f"  Applied {applied} normalizations")

                # Execute discovery queries in parallel
                iter_new = 0
                iter_dups = 0
                if discovery_queries:
                    remaining = p1_budget - p1_used
                    queries_to_run = discovery_queries[:remaining]

                    self._emit(f"\nDiscovery queries ({len(queries_to_run)}):")
                    for i, q in enumerate(queries_to_run, 1):
                        self._emit(f"  {i}. [{q.get('goal','')}] {q.get('search_query','')}", dim=True)

                    results = await self._execute_discoveries(
                        record_set=record_set,
                        queries=queries_to_run,
                        concurrency=concurrency,
                    )
                    p1_used += len(queries_to_run)
                    _emit_progress("Phase 1: Broad discovery", p1_used, p1_budget)

                    self._emit(f"\n  Results:")
                    iter_errors = 0
                    for r in results:
                        new = r.get("new_entities", 0)
                        dups = r.get("duplicates", 0)
                        err = r.get("error", "")
                        focus = r.get("focus", "")[:60]
                        status = f"+{new} new, {dups} dups"
                        if err:
                            status += f" [ERROR: {err[:40]}]"
                            iter_errors += 1
                        self._emit(f"    {focus}  →  {status}")
                        iter_new += new
                        iter_dups += dups

                        # Track zero-yield queries
                        if new == 0 and not err:
                            zero_yield_queries.append(r.get("focus", ""))

                        # Update search angle tracking
                        self._update_angle_tracking(
                            search_angles, r.get("focus", ""), new
                        )

                    if iter_errors == len(results) and len(results) > 0:
                        consecutive_all_timeout_iters += 1
                        if consecutive_all_timeout_iters >= 2:
                            self._emit("\n🛑 Aborting: 2 consecutive all-failure iterations.", bold=True)
                            break
                    else:
                        consecutive_all_timeout_iters = 0

                # Align records with schema
                schema_names = {a.name for a in record_set.schema_attributes}
                self.resolution._align_records_with_schema(record_set, schema_names)

                # Record iteration metrics
                elapsed = (datetime.now() - iter_start).total_seconds()
                iter_record = {
                    "iteration": iteration,
                    "phase": "discovery",
                    "reasoning": reasoning,
                    "discovery_queries": len(discovery_queries),
                    "new_entities": iter_new,
                    "duplicates": iter_dups,
                    "completions_run": 0,
                    "total_entities": len(record_set.records),
                    "queries_used": p1_used,
                    "cost": self.llm.total_cost,
                    "focus": [q.get("search_query", "")[:80] for q in discovery_queries],
                    "duration_s": elapsed,
                }
                history.append(iter_record)

                self._emit(
                    f"\n✓ P1 iter {iteration}: {elapsed:.1f}s, +{iter_new} new, "
                    f"{iter_dups} dups → {len(record_set.records)} entities  "
                    f"[{p1_used}/{p1_budget} queries, ${self.llm.total_cost:.2f}]",
                    bold=True,
                )

                if output_dir:
                    self._save_snapshot(record_set, output_dir, f"p1_{iteration}")

                if should_stop:
                    break

            total_queries_used += p1_used
            self._emit(f"\nPhase 1 complete: {len(record_set.records)} entities, {p1_used} queries used", bold=True)

            # ══════════════════════════════════════════════════════════
            # PHASE 2: Targeted Discovery
            # ══════════════════════════════════════════════════════════
            # Redistribute unused Phase 1 budget to Phase 2
            p2_budget += (p1_budget - p1_used)

            self._emit(f"\n{'═'*60}", bold=True)
            self._emit(f"PHASE 2: TARGETED DISCOVERY (budget: {p2_budget} queries)", bold=True)
            self._emit(f"{'═'*60}")

            p2_used = 0
            p2_iteration = 0

            while p2_used < p2_budget:
                p2_iteration += 1
                iteration += 1
                iter_start = datetime.now()

                self._emit(f"\n{'─'*60}", bold=True)
                self._emit(
                    f"P2 ITERATION {p2_iteration}  │  {p2_used}/{p2_budget} queries  │  "
                    f"${self.llm.total_cost:.2f}  │  {len(record_set.records)} entities",
                    bold=True,
                )

                plan = await self._get_plan(
                    record_set=record_set,
                    history=history,
                    queries_used=p2_used,
                    max_queries=p2_budget,
                    phase="targeted",
                    search_angles=search_angles,
                    zero_yield_queries=zero_yield_queries,
                )

                reasoning = plan.get("reasoning", "")
                discovery_queries = plan.get("queries", [])[:10]
                normalizations = plan.get("normalizations", [])
                completions = plan.get("completions", [])[:5]
                schema_changes = plan.get("schema_changes", [])[:5]
                should_stop = plan.get("stop", False)

                self._emit(f"\n🧠 Agent reasoning:")
                for line in reasoning.split("\n"):
                    self._emit(f"   {line}")
                self._emit(
                    f"\n→ Plan: {len(discovery_queries)} queries, "
                    f"{len(completions)} completions, "
                    f"{len(normalizations)} normalizations"
                    f"{' [STOP signalled]' if should_stop else ''}"
                )

                if should_stop and not discovery_queries and not completions:
                    self._emit("\nAgent decided to stop Phase 2.", bold=True)
                    break

                # Apply schema changes & normalizations
                if schema_changes:
                    applied = self._apply_schema_changes(record_set, schema_changes)
                    self._emit(f"  Applied {applied} schema changes")
                if normalizations:
                    applied = self._apply_normalizations(record_set, normalizations)
                    self._emit(f"  Applied {applied} normalizations")

                # Execute discovery queries in parallel
                iter_new = 0
                iter_dups = 0
                if discovery_queries:
                    remaining = p2_budget - p2_used
                    queries_to_run = discovery_queries[:remaining]

                    self._emit(f"\nDiscovery queries ({len(queries_to_run)}):")
                    for i, q in enumerate(queries_to_run, 1):
                        self._emit(f"  {i}. [{q.get('goal','')}] {q.get('search_query','')}", dim=True)

                    results = await self._execute_discoveries(
                        record_set=record_set,
                        queries=queries_to_run,
                        concurrency=concurrency,
                    )
                    p2_used += len(queries_to_run)
                    _emit_progress("Phase 2: Targeted discovery", p2_used, p2_budget)

                    self._emit(f"\n  Results:")
                    for r in results:
                        new = r.get("new_entities", 0)
                        dups = r.get("duplicates", 0)
                        err = r.get("error", "")
                        focus = r.get("focus", "")[:60]
                        status = f"+{new} new, {dups} dups"
                        if err:
                            status += f" [ERROR: {err[:40]}]"
                        self._emit(f"    {focus}  →  {status}")
                        iter_new += new
                        iter_dups += dups
                        if new == 0 and not err:
                            zero_yield_queries.append(r.get("focus", ""))
                        self._update_angle_tracking(search_angles, r.get("focus", ""), new)

                # Run light completions (parallel)
                completions_run = 0
                if completions and p2_used < p2_budget:
                    remaining = p2_budget - p2_used
                    completions_to_run = completions[:remaining]
                    completions_run, comp_queries = await self._execute_completions_parallel(
                        record_set=record_set,
                        completions=completions_to_run,
                        concurrency=concurrency,
                    )
                    p2_used += comp_queries
                    _emit_progress("Phase 2: Targeted discovery", p2_used, p2_budget)

                # Align records
                schema_names = {a.name for a in record_set.schema_attributes}
                self.resolution._align_records_with_schema(record_set, schema_names)

                elapsed = (datetime.now() - iter_start).total_seconds()
                iter_record = {
                    "iteration": iteration,
                    "phase": "targeted",
                    "reasoning": reasoning,
                    "discovery_queries": len(discovery_queries),
                    "new_entities": iter_new,
                    "duplicates": iter_dups,
                    "completions_run": completions_run,
                    "total_entities": len(record_set.records),
                    "queries_used": p2_used,
                    "cost": self.llm.total_cost,
                    "focus": [q.get("search_query", "")[:80] for q in discovery_queries],
                    "duration_s": elapsed,
                }
                history.append(iter_record)

                self._emit(
                    f"\n✓ P2 iter {p2_iteration}: {elapsed:.1f}s, +{iter_new} new, "
                    f"{completions_run} completions → {len(record_set.records)} entities  "
                    f"[{p2_used}/{p2_budget} queries, ${self.llm.total_cost:.2f}]",
                    bold=True,
                )

                if output_dir:
                    self._save_snapshot(record_set, output_dir, f"p2_{p2_iteration}")

                if should_stop:
                    break

            total_queries_used += p2_used
            self._emit(f"\nPhase 2 complete: {len(record_set.records)} entities, {p2_used} queries used", bold=True)

            # ══════════════════════════════════════════════════════════
            # PHASE 3: Completion
            # ══════════════════════════════════════════════════════════
            # Redistribute unused Phase 2 budget
            p3_budget += (p2_budget - p2_used)

            self._emit(f"\n{'═'*60}", bold=True)
            self._emit(f"PHASE 3: COMPLETION (budget: {p3_budget} queries)", bold=True)
            self._emit(f"{'═'*60}")

            p3_used = 0
            p3_iteration = 0

            while p3_used < p3_budget:
                p3_iteration += 1
                iteration += 1
                iter_start = datetime.now()

                self._emit(f"\n{'─'*60}", bold=True)
                self._emit(
                    f"P3 ITERATION {p3_iteration}  │  {p3_used}/{p3_budget} queries  │  "
                    f"${self.llm.total_cost:.2f}  │  {len(record_set.records)} entities",
                    bold=True,
                )

                plan = await self._get_plan(
                    record_set=record_set,
                    history=history,
                    queries_used=p3_used,
                    max_queries=p3_budget,
                    phase="completion",
                    search_angles=search_angles,
                    zero_yield_queries=zero_yield_queries,
                )

                reasoning = plan.get("reasoning", "")
                discovery_queries = plan.get("queries", [])[:3]
                normalizations = plan.get("normalizations", [])
                completions = plan.get("completions", [])[:30]
                schema_changes = plan.get("schema_changes", [])[:5]
                should_stop = plan.get("stop", False)

                self._emit(f"\n🧠 Agent reasoning:")
                for line in reasoning.split("\n"):
                    self._emit(f"   {line}")
                self._emit(
                    f"\n→ Plan: {len(discovery_queries)} queries, "
                    f"{len(completions)} completions, "
                    f"{len(normalizations)} normalizations"
                    f"{' [STOP signalled]' if should_stop else ''}"
                )

                if should_stop and not discovery_queries and not completions:
                    self._emit("\nAgent decided to stop Phase 3.", bold=True)
                    break

                # Apply schema changes & normalizations
                if schema_changes:
                    applied = self._apply_schema_changes(record_set, schema_changes)
                    self._emit(f"  Applied {applied} schema changes")
                if normalizations:
                    applied = self._apply_normalizations(record_set, normalizations)
                    self._emit(f"  Applied {applied} normalizations")

                # Light discovery (parallel)
                iter_new = 0
                iter_dups = 0
                if discovery_queries and p3_used < p3_budget:
                    remaining = p3_budget - p3_used
                    queries_to_run = discovery_queries[:remaining]

                    self._emit(f"\nDiscovery queries ({len(queries_to_run)}):")
                    for i, q in enumerate(queries_to_run, 1):
                        self._emit(f"  {i}. [{q.get('goal','')}] {q.get('search_query','')}", dim=True)

                    results = await self._execute_discoveries(
                        record_set=record_set,
                        queries=queries_to_run,
                        concurrency=concurrency,
                    )
                    p3_used += len(queries_to_run)
                    _emit_progress("Phase 3: Completion", p3_used, p3_budget)

                    for r in results:
                        new = r.get("new_entities", 0)
                        dups = r.get("duplicates", 0)
                        err = r.get("error", "")
                        focus = r.get("focus", "")[:60]
                        status = f"+{new} new, {dups} dups"
                        if err:
                            status += f" [ERROR: {err[:40]}]"
                        self._emit(f"    {focus}  →  {status}")
                        iter_new += new
                        iter_dups += dups

                # Run completions in PARALLEL
                completions_run = 0
                if completions and p3_used < p3_budget:
                    remaining = p3_budget - p3_used
                    completions_to_run = completions[:remaining]
                    completions_run, comp_queries = await self._execute_completions_parallel(
                        record_set=record_set,
                        completions=completions_to_run,
                        concurrency=concurrency,
                    )
                    p3_used += comp_queries
                    _emit_progress("Phase 3: Completion", p3_used, p3_budget)

                # Align records
                schema_names = {a.name for a in record_set.schema_attributes}
                self.resolution._align_records_with_schema(record_set, schema_names)

                elapsed = (datetime.now() - iter_start).total_seconds()
                iter_record = {
                    "iteration": iteration,
                    "phase": "completion",
                    "reasoning": reasoning,
                    "discovery_queries": len(discovery_queries),
                    "new_entities": iter_new,
                    "duplicates": iter_dups,
                    "completions_run": completions_run,
                    "total_entities": len(record_set.records),
                    "queries_used": p3_used,
                    "cost": self.llm.total_cost,
                    "focus": [q.get("search_query", "")[:80] for q in discovery_queries],
                    "duration_s": elapsed,
                }
                history.append(iter_record)

                self._emit(
                    f"\n✓ P3 iter {p3_iteration}: {elapsed:.1f}s, +{iter_new} new, "
                    f"{completions_run} completions → {len(record_set.records)} entities  "
                    f"[{p3_used}/{p3_budget} queries, ${self.llm.total_cost:.2f}]",
                    bold=True,
                )

                if output_dir:
                    self._save_snapshot(record_set, output_dir, f"p3_{p3_iteration}")

                if should_stop:
                    break

            total_queries_used += p3_used

            # ── Final ──
            self._emit(f"\n{'='*60}", bold=True)
            self._emit(
                f"Run complete: {len(record_set.records)} entities, "
                f"{total_queries_used} queries, ${self.llm.total_cost:.2f}",
                bold=True,
            )
            self._emit(f"{'='*60}")

            if output_dir:
                self._save_snapshot(record_set, output_dir, "final")
                self.llm.save_all_responses()
                usage = self.llm.get_usage_stats()
                with open(os.path.join(output_dir, "usage_stats.json"), "w") as f:
                    json.dump(usage, f, indent=2)
                with open(os.path.join(output_dir, "agentic_history.json"), "w") as f:
                    json.dump(history, f, indent=2, default=str)
                self._emit(f"Saved to {output_dir}/", dim=True)

        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None

        return history

    # ------------------------------------------------------------------
    # Agent planning call
    # ------------------------------------------------------------------

    async def _get_plan(
        self,
        record_set: RecordSet,
        history: list[dict],
        queries_used: int,
        max_queries: int,
        phase: str = "discovery",
        search_angles: list[dict] | None = None,
        zero_yield_queries: list[str] | None = None,
    ) -> dict:
        """Ask the reasoning model to plan the next iteration.

        Args:
            phase: One of "discovery", "targeted", "completion"
            search_angles: Tracked search angles for Phase 1/2
            zero_yield_queries: Free-form queries that found nothing
        """
        max_budget = self.config.max_budget or 999.0

        # Select phase-specific prompt
        if phase == "discovery":
            prompt_template = PHASE1_DISCOVERY_PROMPT
        elif phase == "targeted":
            prompt_template = PHASE2_TARGETED_PROMPT
        else:
            prompt_template = PHASE3_COMPLETION_PROMPT

        # Build common variables
        from datetime import date as _date
        variables = {
            "category": record_set.category,
            "guidance": record_set.guidance or "(none)",
            "schema_summary": build_schema_summary(record_set),
            "entity_count": len(record_set.records),
            "state_summary": build_state_summary(record_set),
            "queries_used": queries_used,
            "max_queries": max_queries,
            "cost": self.llm.total_cost,
            "max_budget": max_budget,
            "current_date": _date.today().isoformat(),
        }

        # Phase-specific variables
        if phase == "discovery":
            variables["search_angles"] = format_search_angles(search_angles or [])
            variables["zero_yield_summary"] = format_zero_yield_summary(
                zero_yield_queries or [], search_angles
            )
            variables["history_window"] = min(15, len(history))
            variables["total_queries"] = queries_used
            variables["query_history"] = build_query_history(history)
        elif phase == "targeted":
            variables["productive_angles"] = format_productive_angles(search_angles or [])
            variables["zero_yield_summary"] = format_zero_yield_summary(
                zero_yield_queries or [], search_angles
            )
            variables["history_window"] = min(15, len(history))
            variables["total_queries"] = queries_used
            variables["query_history"] = build_query_history(history)

        prompt = prompt_template.format(**variables)

        # Use low reasoning for completion, medium for discovery/targeted
        original_effort = self.config.reasoning_effort
        self.config.reasoning_effort = "low" if phase == "completion" else "medium"
        self.llm.set_progress_context(f"Planning {phase} iteration")

        try:
            result = await self.llm.structured_completion(
                prompt=prompt,
                response_format=AGENT_PLAN_SCHEMA,
            )
        finally:
            self.config.reasoning_effort = original_effort
            self.llm.set_progress_context("")

        return result

    # ------------------------------------------------------------------
    # Discovery execution
    # ------------------------------------------------------------------

    async def _execute_discoveries(
        self,
        record_set: RecordSet,
        queries: list[dict],
        concurrency: int,
    ) -> list[dict]:
        """Execute agent-issued discovery queries in parallel."""
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        results_lock = asyncio.Lock()

        async def run_one(idx: int, query_spec: dict):
            async with semaphore:
                search_query = query_spec.get("search_query", "")
                goal = query_spec.get("goal", "")
                self.llm.set_progress_context(
                    f"Discovery {idx + 1}/{len(queries)}: {goal}"
                )

                try:
                    new_records = await asyncio.wait_for(
                        self.extraction.discover_entities(
                            record_set,
                            subcategory_focus=search_query,
                        ),
                        timeout=600,  # 10-min hard cap per discovery query
                    )
                except Exception as e:
                    logger.error(f"Discovery query failed: {e}")
                    async with results_lock:
                        results.append({
                            "focus": search_query,
                            "new_entities": 0,
                            "duplicates": 0,
                            "error": str(e),
                        })
                    return

                # Deduplicate against existing records
                dups = 0
                if new_records and record_set.records:
                    new_labels = [r.label for r in new_records]
                    existing_labels = record_set.get_labels()
                    alias_map = {
                        r.label: r.aliases
                        for r in record_set.records if r.aliases
                    }
                    dup_map = await self.resolution.check_duplicates(
                        new_labels, existing_labels,
                        use_fuzzy=True,
                        existing_aliases=alias_map,
                    )
                    # Merge duplicates
                    for rec in new_records:
                        if rec.label in dup_map:
                            existing = record_set.get_record(dup_map[rec.label])
                            if existing:
                                existing.merge_from(rec)
                                dups += 1
                    new_records = [r for r in new_records if r.label not in dup_map]

                # Add new records
                added = 0
                for rec in new_records:
                    was_added, existing = record_set.add_record(rec, use_fuzzy=True)
                    if was_added:
                        added += 1
                    elif existing:
                        dups += 1

                logger.debug(
                    f"  [{goal}] {search_query[:70]}... → "
                    f"+{added} new, {dups} dups"
                )

                async with results_lock:
                    results.append({
                        "focus": search_query,
                        "new_entities": added,
                        "duplicates": dups,
                    })

        tasks = [run_one(i, q) for i, q in enumerate(queries)]
        await asyncio.gather(*tasks)

        self.llm.set_progress_context("")
        return results

    # ------------------------------------------------------------------
    # Parallel completions
    # ------------------------------------------------------------------

    async def _execute_completions_parallel(
        self,
        record_set: RecordSet,
        completions: list[dict],
        concurrency: int,
    ) -> tuple[int, int]:
        """
        Execute agent-issued completions in parallel.

        Returns (completions_run, queries_consumed).
        """
        if not completions:
            return 0, 0

        schema_name_map = {
            a.name.lower(): a.name
            for a in record_set.schema_attributes
        }

        # Pre-resolve and filter completions
        resolved_completions: list[tuple[Record, list[str]]] = []
        for comp in completions:
            entity_label = comp.get("entity", "")
            raw_attrs = comp.get("missing_attributes", [])
            resolved_attrs = []
            for ra in raw_attrs:
                matched = schema_name_map.get(ra.lower())
                resolved_attrs.append(matched if matched else ra)
            record = record_set.get_record(entity_label)
            if not record or not resolved_attrs:
                continue
            actionable = [
                a for a in resolved_attrs
                if record.completion_attempts.get(a, 0) < 2
            ]
            if not actionable:
                self._emit(f"  {entity_label}: all attrs exhausted, skipping", dim=True)
                continue
            self._emit(f"  {entity_label}: {actionable}", dim=True)
            resolved_completions.append((record, actionable))

        if not resolved_completions:
            return 0, 0

        self._emit(f"\nRunning {len(resolved_completions)} completions in parallel (concurrency={concurrency})...")

        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0
        completed_lock = asyncio.Lock()

        async def run_one_completion(record: Record, actionable: list[str]):
            nonlocal completed_count
            async with semaphore:
                self.llm.set_progress_context(f"Completion: {record.label}")
                missing_before = {
                    a for a in actionable
                    if not record.attributes.get(a) or not record.attributes[a].value
                }
                try:
                    await asyncio.wait_for(
                        self.extraction.expand_record(
                            record, record_set,
                            target_attributes=actionable,
                        ),
                        timeout=600,
                    )
                    async with completed_lock:
                        completed_count += 1
                    # Track attempts for attrs still missing after search
                    for a in missing_before:
                        av = record.attributes.get(a)
                        if not av or not av.value:
                            record.completion_attempts[a] = record.completion_attempts.get(a, 0) + 1
                except Exception as e:
                    self._emit(f"  ⚠ {record.label} failed: {e}")
                    for a in actionable:
                        if not record.attributes.get(a) or not record.attributes[a].value:
                            record.completion_attempts[a] = record.completion_attempts.get(a, 0) + 1

        tasks = [
            run_one_completion(rec, attrs)
            for rec, attrs in resolved_completions
        ]
        await asyncio.gather(*tasks)
        self.llm.set_progress_context("")

        self._emit(f"  Completed {completed_count}/{len(resolved_completions)} successfully")
        return completed_count, len(resolved_completions)

    # ------------------------------------------------------------------
    # Search angle tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _update_angle_tracking(
        search_angles: list[dict],
        query_focus: str,
        new_entities: int,
    ):
        """
        Update search angle tracking when a query result comes back.

        Matches the query focus against known angles to track yield.
        Also marks angles as 'used' if the query closely matches.
        """
        if not query_focus or not search_angles:
            return
        focus_lower = query_focus.lower()
        for angle in search_angles:
            # Check if this query was based on this angle
            val_lower = angle["value"].lower()
            attr_lower = angle["attribute"].lower()
            if val_lower in focus_lower and attr_lower in focus_lower:
                angle["used"] = True
                angle["yield"] = angle.get("yield", 0) + new_entities
                break

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _apply_normalizations(
        self,
        record_set: RecordSet,
        normalizations: list[dict],
    ) -> int:
        """
        Apply agent-issued value normalizations to the record set.

        Each normalization merges multiple values into a canonical form
        for a specific attribute.

        Returns count of values changed.
        """
        changes = 0
        schema_attr_names = {a.name for a in record_set.schema_attributes}

        for norm in normalizations:
            attr_name = norm.get("attribute", "")
            merge_values = set(norm.get("merge_values", []))
            canonical = norm.get("canonical", "")

            if not attr_name or not merge_values or not canonical:
                continue
            if attr_name not in schema_attr_names:
                continue

            # Apply to all records (both attributes and additional_attributes)
            for record in record_set.records:
                for attr_dict in (record.attributes, record.additional_attributes):
                    av = attr_dict.get(attr_name)
                    if not av or not av.values:
                        continue

                    # Update the underlying SourcedValue objects
                    record_changed = False
                    for sv in av.values:
                        if sv.value in merge_values and sv.value != canonical:
                            sv.value = canonical
                            record_changed = True
                    if record_changed:
                        changes += 1

            # Also update provisional values on the schema attribute
            for attr in record_set.schema_attributes:
                if attr.name == attr_name and attr.provisional_values:
                    new_prov = []
                    seen = set()
                    for v in attr.provisional_values:
                        norm_v = canonical if v in merge_values else v
                        if norm_v not in seen:
                            seen.add(norm_v)
                            new_prov.append(norm_v)
                    attr.provisional_values = new_prov

        if changes:
            logger.info(f"Normalized {changes} attribute values")
        return changes

    # ------------------------------------------------------------------
    # Schema changes
    # ------------------------------------------------------------------

    def _apply_schema_changes(
        self,
        record_set: RecordSet,
        schema_changes: list[dict],
    ) -> int:
        """
        Apply agent-issued schema curation changes.

        Supported kinds:
        - demote: Remove attribute from core schema (values → additional_attributes)
        - promote: Add attribute to core schema (values ← additional_attributes)
        - rename: Rename an attribute across all records
        - decompose: Split a compound value into atomic pieces

        Returns count of changes applied.
        """
        applied = 0
        schema_attr_names = {a.name for a in record_set.schema_attributes}

        for change in schema_changes:
            kind = change.get("kind", "")
            attr_name = change.get("attribute", "")
            reason = change.get("reason", "")

            if kind == "demote" and attr_name in schema_attr_names:
                # Move attribute from schema to additional_attributes
                record_set.schema_attributes = [
                    a for a in record_set.schema_attributes
                    if a.name != attr_name
                ]
                for record in record_set.records:
                    if attr_name in record.attributes:
                        av = record.attributes.pop(attr_name)
                        record.additional_attributes[attr_name] = av
                schema_attr_names.discard(attr_name)
                self._emit(f"  ↓ Demoted '{attr_name}': {reason}", dim=True)
                applied += 1

            elif kind == "promote" and attr_name not in schema_attr_names:
                # Count how many records have this in additional_attributes
                count = sum(
                    1 for r in record_set.records
                    if r.additional_attributes.get(attr_name)
                    and r.additional_attributes[attr_name].value
                )
                if count >= 2:  # Need at least some presence
                    new_attr = SchemaAttribute(
                        name=attr_name,
                        frequency=count / len(record_set.records) if record_set.records else 0,
                    )
                    record_set.schema_attributes.append(new_attr)
                    schema_attr_names.add(attr_name)
                    # Move values from additional to schema attributes
                    for record in record_set.records:
                        if attr_name in record.additional_attributes:
                            av = record.additional_attributes.pop(attr_name)
                            record.attributes[attr_name] = av
                    self._emit(f"  ↑ Promoted '{attr_name}' ({count} records): {reason}", dim=True)
                    applied += 1
                else:
                    self._emit(f"  ⚠ Skipped promote '{attr_name}' — only {count} records have it", dim=True)

            elif kind == "rename" and attr_name in schema_attr_names:
                new_name = change.get("new_name", "")
                if new_name and new_name != attr_name and new_name not in schema_attr_names:
                    # Rename schema attribute
                    for a in record_set.schema_attributes:
                        if a.name == attr_name:
                            a.name = new_name
                            break
                    # Rename in all records (both attributes and additional_attributes)
                    for record in record_set.records:
                        if attr_name in record.attributes:
                            record.attributes[new_name] = record.attributes.pop(attr_name)
                        if attr_name in record.additional_attributes:
                            record.additional_attributes[new_name] = record.additional_attributes.pop(attr_name)
                    schema_attr_names.discard(attr_name)
                    schema_attr_names.add(new_name)
                    self._emit(f"  ✏ Renamed '{attr_name}' → '{new_name}': {reason}", dim=True)
                    applied += 1

            elif kind == "decompose":
                compound_val = change.get("compound_value", "")
                replacements = change.get("replacements", [])
                if attr_name and compound_val and replacements:
                    decomp_count = self.resolution.decompose_compound_values(
                        record_set,
                        [{"source_attribute": attr_name, "compound_value": compound_val, "replacements": replacements}],
                    )
                    pieces = [f"{r.get('attribute')}={r.get('value')}" for r in replacements]
                    self._emit(
                        f"  ✂ Decomposed '{compound_val}' in '{attr_name}' → "
                        f"{pieces} "
                        f"({decomp_count} records): {reason}",
                        dim=True,
                    )
                    applied += 1

        return applied

    # ------------------------------------------------------------------
    # State restoration & seed ingestion
    # ------------------------------------------------------------------

    def _restore_state(
        self,
        record_set: RecordSet,
        filepath: str,
    ) -> dict:
        """
        Restore a Schemify save file as the **foundation state**.

        This replaces the current ``schema_attributes`` on *record_set*
        with those from the save file, and merges all saved records.
        Use this when resuming a previous run or when the save file
        defines the canonical schema that additional seed data should
        conform to.

        Args:
            record_set: The live RecordSet to update.
            filepath: Path to a Schemify JSON export (must contain
                a ``category`` key — the output of ``Schemify.save()``).

        Returns:
            Dict with ``records`` (int added) and ``schema_attrs``
            (int attributes restored).
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "category" not in data:
            raise ValueError(
                f"seed_state file must be a Schemify JSON export "
                f"(dict with 'category' key). Got: {filepath}"
            )

        # --- Restore schema attributes (foundation) ---
        saved_attrs = [
            SchemaAttribute(
                name=a["name"],
                description=a.get("description"),
                required=a.get("required", False),
                frequency=a.get("frequency", 0.0),
                is_multi_valued=a.get("is_multi_valued", True),
                is_closed_set=a.get("is_closed_set", False),
                provisional_values=[
                    pv if isinstance(pv, str) else pv.get("value", str(pv))
                    for pv in a.get("provisional_values", [])
                ],
            )
            for a in data.get("schema_attributes", [])
        ]
        if saved_attrs:
            record_set.schema_attributes = saved_attrs
            logger.info(
                f"Restored {len(saved_attrs)} schema attributes from {filepath}"
            )

        # --- Restore records ---
        added = 0
        for r in data.get("records", []):
            rec = Record.from_dict(r)
            was_new, _ = record_set.add_record(rec, use_fuzzy=True)
            if was_new:
                added += 1

        # --- Prune schema bloat ---
        # Seed files from previous runs may carry hundreds of 0%-fill
        # schema attributes (entity-specific one-offs). Strip any that
        # have zero actual data across restored records, keeping only
        # attributes that at least one record has a value for.
        if record_set.records:
            schema_before = len(record_set.schema_attributes)
            record_set.schema_attributes = [
                a for a in record_set.schema_attributes
                if any(
                    a.name in r.attributes and r.attributes[a.name].value
                    for r in record_set.records
                )
            ]
            pruned = schema_before - len(record_set.schema_attributes)
            if pruned:
                logger.info(
                    f"Pruned {pruned} zero-fill schema attrs "
                    f"({schema_before} → {len(record_set.schema_attributes)})"
                )
                # Move pruned attrs from record.attributes → additional_attributes
                kept_names = {a.name for a in record_set.schema_attributes}
                for rec in record_set.records:
                    to_demote = [
                        (k, v) for k, v in rec.attributes.items()
                        if k not in kept_names
                    ]
                    for k, v in to_demote:
                        rec.additional_attributes[k] = rec.attributes.pop(k)

        return {"records": added, "schema_attrs": len(record_set.schema_attributes)}

    async def _ingest_seed_records(
        self,
        record_set: RecordSet,
        seed_records: "str | list[dict] | Any",
    ) -> int:
        """
        Ingest prior data into *record_set* before the exploration loop.

        If the seed data has columns that don't match the current schema,
        the LLM is called to auto-infer attribute and value mappings based
        on column names *and* sample value distributions.

        Accepts three formats:
        - **str** – path to a JSON file.  If the JSON contains a ``category``
          key it is treated as a Schemify export; otherwise a list of flat dicts.
        - **list[dict]** – flat dicts with an ``Entity`` or ``label`` key.
        - **DataFrame** – converted to ``list[dict]``.

        Returns the number of records successfully added or merged.
        """
        rows: list[dict] | None = None
        native_records: list[Record] | None = None

        # --- resolve input to rows or native_records ---
        if isinstance(seed_records, str):
            with open(seed_records, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "category" in data:
                native_records = [
                    Record.from_dict(r) for r in data.get("records", [])
                ]
            elif isinstance(data, list):
                rows = data
            else:
                raise ValueError(
                    "Seed JSON must be a Schemify export (dict with 'category') "
                    "or a plain list of dicts."
                )
        elif isinstance(seed_records, list):
            rows = seed_records
        else:
            # Assume DataFrame-like (has .to_dict)
            rows = seed_records.to_dict(orient="records")

        # --- auto-detect schema mismatch and remap via LLM ---
        schema_attr_names = {a.name for a in record_set.schema_attributes}
        column_map: dict[str, str] | None = None
        value_map: dict[str, dict[str, str]] | None = None

        if rows is not None:
            seed_columns = {
                k for row in rows for k in row.keys()
            } - {"Entity", "label", "name", "aliases", "_confidence"}
            unmatched = seed_columns - schema_attr_names
            if unmatched:
                remap = await self._auto_remap_schema(
                    record_set, rows, unmatched,
                )
                column_map = remap.get("column_map")
                value_map = remap.get("value_map")
        elif native_records is not None:
            seed_columns: set[str] = set()
            for rec in native_records:
                seed_columns.update(rec.attributes.keys())
                seed_columns.update(rec.additional_attributes.keys())
            unmatched = seed_columns - schema_attr_names
            if unmatched:
                # Convert native records to sample rows for the LLM
                sample_rows = self._records_to_sample_rows(native_records)
                remap = await self._auto_remap_schema(
                    record_set, sample_rows, unmatched,
                )
                column_map = remap.get("column_map")
                value_map = remap.get("value_map")

        # --- apply remapping to native_records (Schemify format) ---
        if native_records is not None and (column_map or value_map):
            native_records = self._remap_records(
                native_records, column_map, value_map,
            )

        # --- convert flat rows into Record objects ---
        if rows is not None:
            if column_map or value_map:
                rows = self._remap_rows(rows, column_map, value_map)

            native_records = []
            for row in rows:
                label = row.get("Entity") or row.get("label") or row.get("name")
                if not label:
                    continue
                rec = Record(label=str(label))
                for key, val in row.items():
                    if key in ("Entity", "label", "name", "aliases", "_confidence"):
                        continue
                    if val is None or (isinstance(val, float) and val != val):
                        continue  # skip NaN / None
                    is_schema = key in schema_attr_names
                    av = AttributeValue()
                    av.add_value(str(val))
                    rec.set_attribute(
                        key,
                        av,
                        is_schema_attr=is_schema,
                    )
                native_records.append(rec)

        # --- merge into record_set with fuzzy dedup ---
        added = 0
        for rec in native_records or []:
            was_new, _ = record_set.add_record(rec, use_fuzzy=True)
            if was_new:
                added += 1

        return added

    # ------------------------------------------------------------------
    # Auto schema remapping via LLM
    # ------------------------------------------------------------------

    async def _auto_remap_schema(
        self,
        record_set: RecordSet,
        seed_rows: list[dict],
        unmatched_columns: set[str],
    ) -> dict:
        """
        Use the LLM to infer column and value mappings from seed data
        that doesn't match the current schema.

        Sends both attribute names AND sample values from each side so
        the model can reason about semantic equivalence even when names
        differ (e.g. "Tool Category" → "Tool type").

        Returns:
            ``{"column_map": {...}, "value_map": {...}}``
        """
        from collections import Counter

        # Build target schema summary with sample values
        target_attrs: list[dict] = []
        for attr in record_set.schema_attributes:
            sample_vals = attr.provisional_values[:15] if attr.provisional_values else []
            # Also gather observed values from existing records
            if not sample_vals:
                observed = [
                    rec.attributes[attr.name].value
                    for rec in record_set.records
                    if attr.name in rec.attributes and rec.attributes[attr.name].value
                ]
                sample_vals = [v for v, _ in Counter(observed).most_common(15)]
            target_attrs.append({
                "name": attr.name,
                "description": attr.description or "",
                "sample_values": sample_vals,
            })

        # Build source column summary with sample values
        source_attrs: list[dict] = []
        for col in sorted(unmatched_columns):
            observed = [
                str(row[col]) for row in seed_rows
                if col in row and row[col] is not None
                and not (isinstance(row[col], float) and row[col] != row[col])
            ]
            sample_vals = [v for v, _ in Counter(observed).most_common(15)]
            source_attrs.append({
                "name": col,
                "sample_values": sample_vals,
            })

        prompt = """You are mapping columns from a seed dataset onto a target schema.

The seed data has columns that don't match the target schema by name.
For each unmatched source column, decide:
1. Which target attribute it maps to (based on BOTH name similarity AND value overlap)
2. How individual values should be remapped to match the target vocabulary

If a source column has no reasonable match in the target schema, map it to null.

## Target Schema Attributes

{target_schema}

## Unmatched Source Columns (with sample values)

{source_columns}

## Instructions

- Compare column names AND value distributions to find the best match
- A column named "Region" with values ["US", "UK", "EU"] likely maps to a geographic attribute
- A column named "Type" with values ["NGO", "For-profit"] likely maps to an organization type attribute
- For value_mappings, only include values that need remapping — skip values that already match
- If values are close but use different conventions (e.g. "Law Enforcement" vs "Law enforcement"), include a mapping
- If a source column genuinely doesn't fit any target attribute, set target_attribute to null"""

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_remap",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief analysis of how source columns map to target attributes",
                        },
                        "mappings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_column": {
                                        "type": "string",
                                        "description": "Name of the unmatched source column",
                                    },
                                    "target_attribute": {
                                        "type": ["string", "null"],
                                        "description": "Name of the matching target schema attribute, or null if no match",
                                    },
                                    "value_mappings": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "from": {"type": "string"},
                                                "to": {"type": "string"},
                                            },
                                            "required": ["from", "to"],
                                            "additionalProperties": False,
                                        },
                                        "description": "Value remappings (only for values that differ)",
                                    },
                                },
                                "required": ["source_column", "target_attribute", "value_mappings"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["reasoning", "mappings"],
                    "additionalProperties": False,
                },
            },
        }

        result = await self.llm.structured_completion(
            prompt=prompt,
            response_format=schema,
            variables={
                "target_schema": json.dumps(target_attrs, indent=2),
                "source_columns": json.dumps(source_attrs, indent=2),
            },
        )

        # Parse LLM response into column_map + value_map
        column_map: dict[str, str] = {}
        value_map: dict[str, dict[str, str]] = {}

        for m in result.get("mappings", []):
            src = m["source_column"]
            tgt = m["target_attribute"]
            if tgt is None:
                continue  # No match — will go to additional_attributes
            column_map[src] = tgt
            val_mappings = {vm["from"]: vm["to"] for vm in m.get("value_mappings", [])}
            if val_mappings:
                value_map[tgt] = val_mappings

        # Log what the LLM decided
        reasoning = result.get("reasoning", "")
        self._emit(f"  Auto-remap reasoning: {reasoning}", dim=True)
        for src, tgt in column_map.items():
            n_vals = len(value_map.get(tgt, {}))
            val_note = f" ({n_vals} value remaps)" if n_vals else ""
            self._emit(f"  ✎ {src} → {tgt}{val_note}", dim=True)

        unmapped = unmatched_columns - set(column_map.keys())
        if unmapped:
            self._emit(
                f"  ⊘ No match: {', '.join(sorted(unmapped))} → additional_attributes",
                dim=True,
            )

        return {"column_map": column_map or None, "value_map": value_map or None}

    @staticmethod
    def _records_to_sample_rows(
        records: list[Record],
        max_rows: int = 50,
    ) -> list[dict]:
        """Convert Record objects to flat dicts for the remap LLM prompt."""
        rows: list[dict] = []
        for rec in records[:max_rows]:
            row: dict = {"Entity": rec.label}
            for attr_name, av in rec.attributes.items():
                row[attr_name] = av.value if av.value else None
            for attr_name, av in rec.additional_attributes.items():
                row[attr_name] = av.value if av.value else None
            rows.append(row)
        return rows

    @staticmethod
    def _remap_rows(
        rows: list[dict],
        column_map: dict[str, str] | None,
        value_map: dict[str, dict[str, str]] | None,
    ) -> list[dict]:
        """
        Apply column renaming and value remapping to flat dicts.

        Column remap runs first so that value_map keys use *target* names.
        """
        remapped: list[dict] = []
        for row in rows:
            new_row: dict = {}
            for key, val in row.items():
                target_key = column_map.get(key, key) if column_map else key
                if value_map and target_key in value_map and val is not None:
                    val = value_map[target_key].get(str(val), val)
                new_row[target_key] = val
            remapped.append(new_row)
        return remapped

    @staticmethod
    def _remap_records(
        records: list[Record],
        column_map: dict[str, str] | None,
        value_map: dict[str, dict[str, str]] | None,
    ) -> list[Record]:
        """
        Apply column renaming and value remapping to deserialized Records.

        Operates on the Record's ``attributes`` and ``additional_attributes``
        dicts in-place, moving entries to their new key names and replacing
        values.
        """
        for rec in records:
            for attr_dict in (rec.attributes, rec.additional_attributes):
                if column_map:
                    keys_to_remap = [k for k in attr_dict if k in column_map]
                    for old_key in keys_to_remap:
                        new_key = column_map[old_key]
                        av = attr_dict.pop(old_key)
                        if new_key in attr_dict:
                            for sv in av.values:
                                attr_dict[new_key].add_value(sv.value)
                        else:
                            attr_dict[new_key] = av

                if value_map:
                    for attr_name, val_map in value_map.items():
                        if attr_name in attr_dict:
                            av = attr_dict[attr_name]
                            for sv in av.values:
                                if sv.value in val_map:
                                    sv.value = val_map[sv.value]
        return records

    # ------------------------------------------------------------------
    # Verification pass
    # ------------------------------------------------------------------

    async def verify_unverified_entities(
        self,
        record_set: RecordSet,
        concurrency: int = 12,
        output_dir: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """
        Attempt to verify all unverified attribute values via web search.

        For each entity that has at least one unsourced attribute value,
        makes ONE web-search expansion call targeting those attributes.
        Does NOT delete any values or entities — just adds sources where
        the web confirms them.

        Args:
            record_set: The RecordSet to verify.
            concurrency: Max parallel web searches.
            output_dir: If set, logs are written here.

        Returns:
            Dict with verification statistics.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.llm.set_output_dir(output_dir)
            if not self._log_file:
                log_path = os.path.join(output_dir, "agentic_log.txt")
                self._log_file = open(log_path, "a", encoding="utf-8")

        self._emit(f"\n{'='*60}", bold=True)
        self._emit("VERIFICATION PASS", bold=True)
        self._emit(f"{'='*60}")

        schema_names = {a.name for a in record_set.schema_attributes}

        # Identify entities needing verification
        to_verify: list[tuple[Record, list[str]]] = []
        already_verified = 0
        for record in record_set.records:
            unsourced_attrs: list[str] = []
            for attr_name, av in record.attributes.items():
                if attr_name not in schema_names:
                    continue
                if av.value and not av.sources:
                    unsourced_attrs.append(attr_name)
            # Also detect schema attributes entirely missing from the record
            for attr_name in schema_names:
                if attr_name not in record.attributes:
                    unsourced_attrs.append(attr_name)
            if unsourced_attrs:
                to_verify.append((record, unsourced_attrs))
            else:
                # All filled attrs already have sources (or no attrs at all)
                already_verified += 1

        self._emit(
            f"  {len(to_verify)} entities need verification, "
            f"{already_verified} already verified",
        )
        if not to_verify:
            self._emit("  Nothing to verify.", bold=True)
            return {
                "entities_attempted": 0,
                "entities_already_verified": already_verified,
                "attrs_before_unsourced": 0,
                "attrs_after_unsourced": 0,
                "attrs_newly_sourced": 0,
                "errors": 0,
            }

        # Count unsourced attrs before
        attrs_before_unsourced = sum(len(attrs) for _, attrs in to_verify)

        # Run one expand_record call per entity, in parallel. Use a semaphore
        # plus as_completed so a slow web_search call does not stall the
        # whole batch — the pool stays saturated for the full run.
        semaphore = asyncio.Semaphore(concurrency)
        errors = 0
        errors_lock = asyncio.Lock()
        total = len(to_verify)

        async def verify_one(record: Record, unsourced: list[str]):
            nonlocal errors
            async with semaphore:
                self.llm.set_progress_context(f"Verify: {record.label}")
                try:
                    await asyncio.wait_for(
                        self.extraction.expand_record(
                            record, record_set,
                            target_attributes=unsourced,
                        ),
                        timeout=90,
                    )
                except Exception as e:
                    logger.error(f"Verification failed for {record.label}: {e}")
                    async with errors_lock:
                        errors += 1
            return record.label

        tasks = [verify_one(rec, attrs) for rec, attrs in to_verify]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            label = await coro
            completed += 1
            if progress_callback is not None:
                try:
                    progress_callback(completed, total, label)
                except Exception:  # noqa: BLE001
                    pass
            if completed == 1 or completed % max(1, concurrency) == 0 or completed == total:
                self._emit(
                    f"  Verified {completed}/{total} ({label})",
                    dim=True,
                )

        self.llm.set_progress_context("")

        # Recompute confidence (and verified flag) on all touched records
        for record, _ in to_verify:
            for attr_name, av in record.attributes.items():
                if attr_name in schema_names and av.value:
                    av.compute_confidence()

        # Recount unsourced attrs after verification
        attrs_after_unsourced = 0
        for record, _ in to_verify:
            for attr_name, av in record.attributes.items():
                if attr_name in schema_names and av.value and not av.sources:
                    attrs_after_unsourced += 1

        newly_sourced = attrs_before_unsourced - attrs_after_unsourced

        stats = {
            "entities_attempted": len(to_verify),
            "entities_already_verified": already_verified,
            "attrs_before_unsourced": attrs_before_unsourced,
            "attrs_after_unsourced": attrs_after_unsourced,
            "attrs_newly_sourced": newly_sourced,
            "errors": errors,
        }

        self._emit(f"\n{'─'*40}")
        self._emit("Verification results:", bold=True)
        self._emit(f"  Entities attempted:     {stats['entities_attempted']}")
        self._emit(f"  Already verified:       {stats['entities_already_verified']}")
        self._emit(f"  Unsourced attrs before: {stats['attrs_before_unsourced']}")
        self._emit(f"  Unsourced attrs after:  {stats['attrs_after_unsourced']}")
        self._emit(f"  Newly sourced:          {stats['attrs_newly_sourced']}")
        self._emit(f"  Errors:                 {stats['errors']}")
        self._emit(f"{'─'*40}")

        return stats

    # ------------------------------------------------------------------
    # Finalization — produce a high-quality-only dataset
    # ------------------------------------------------------------------

    def finalize(
        self,
        record_set: RecordSet,
        output_dir: str | None = None,
    ) -> RecordSet:
        """
        Build a finalized RecordSet containing only high-quality entities.

        An entity is included only if it has **at least one schema attribute
        with a web source**.  Within each included entity, only attribute
        values that have at least one source are kept; unsourced values are
        dropped.  The original *record_set* is NOT modified.

        If *output_dir* is provided, writes ``final.json`` and ``final.csv``
        to that directory.

        Returns:
            A new RecordSet with only verified entities and values.
        """
        schema_names = {a.name for a in record_set.schema_attributes}

        included: list[Record] = []
        excluded_labels: list[str] = []

        for record in record_set.records:
            # Check if entity has at least one sourced schema attribute
            has_sourced = any(
                av.sources
                for attr_name, av in record.attributes.items()
                if attr_name in schema_names and av.value
            )
            if not has_sourced:
                excluded_labels.append(record.label)
                continue

            # Build a filtered copy — keep only sourced values
            filtered = Record(
                label=record.label,
                aliases=list(record.aliases),
                alias_counts=dict(record.alias_counts),
                created_at=record.created_at,
                updated_at=record.updated_at,
            )

            for attr_name, av in record.attributes.items():
                if av.sources and av.value:
                    filtered.attributes[attr_name] = av
            for attr_name, av in record.additional_attributes.items():
                if av.sources and av.value:
                    filtered.additional_attributes[attr_name] = av

            included.append(filtered)

        # Build the new RecordSet (reuse schema)
        finalized = RecordSet(
            category=record_set.category,
            guidance=record_set.guidance,
            records=included,
            schema_attributes=list(record_set.schema_attributes),
            created_at=record_set.created_at,
        )

        self._emit(f"\n{'='*60}", bold=True)
        self._emit("FINALIZATION", bold=True)
        self._emit(f"{'='*60}")
        self._emit(f"  Total entities:    {len(record_set.records)}")
        self._emit(f"  Included (sourced): {len(included)}")
        self._emit(f"  Excluded (no sources): {len(excluded_labels)}")

        if excluded_labels:
            shown = excluded_labels[:20]
            tail = f" ... +{len(excluded_labels) - 20} more" if len(excluded_labels) > 20 else ""
            self._emit(f"  Excluded: {', '.join(shown)}{tail}", dim=True)

        # Per-attribute fill rates in finalized set
        if included:
            self._emit(f"\n  Finalized attribute fill rates:")
            for attr in record_set.schema_attributes:
                filled = sum(
                    1 for r in included
                    if attr.name in r.attributes and r.attributes[attr.name].value
                )
                pct = filled * 100 // len(included)
                self._emit(f"    {attr.name}: {filled}/{len(included)} ({pct}%)")

        # Save to disk
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            import pandas as pd

            # JSON
            json_path = os.path.join(output_dir, "final.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(finalized.to_dict(), f, indent=2, ensure_ascii=False)

            # CSV (flat)
            schema_attr_names = [a.name for a in finalized.schema_attributes]
            rows = []
            for rec in finalized.records:
                row = {"Entity": rec.label}
                for attr_name in schema_attr_names:
                    av = rec.attributes.get(attr_name)
                    row[attr_name] = av.value if av and av.value else ""
                rows.append(row)
            df = pd.DataFrame(rows)
            csv_path = os.path.join(output_dir, "final.csv")
            df.to_csv(csv_path, index=False)

            # Dashboard JS (slim, for local file:// loading)
            js_path = os.path.join(output_dir, "dashboard_data.js")
            export_dashboard_js(finalized, js_path)

            self._emit(f"\n  Saved finalized dataset to {output_dir}/ (final.json, final.csv, dashboard_data.js)", dim=True)

        return finalized

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _save_snapshot(
        self,
        record_set: RecordSet,
        output_dir: str,
        label: str | int,
    ):
        """Save CSV + JSON snapshot."""
        from .schemify import Schemify  # Local import to avoid circular

        csv_path = os.path.join(output_dir, f"snapshot_{label}.csv")
        json_path = os.path.join(output_dir, f"snapshot_{label}.json")

        # Build DataFrame manually from records
        import pandas as pd
        rows = []
        schema_attrs = [a.name for a in record_set.schema_attributes]
        for record in record_set.records:
            row = {"Entity": record.label}
            for attr_name in schema_attrs:
                av = record.attributes.get(attr_name)
                row[attr_name] = av.value if av and av.value else ""
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # JSON export
        records_data = []
        for record in record_set.records:
            rec_dict = {"label": record.label, "aliases": record.aliases or []}
            for attr_name in schema_attrs:
                av = record.attributes.get(attr_name)
                rec_dict[attr_name] = av.value if av and av.value else ""
            records_data.append(rec_dict)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Snapshot saved: {csv_path}")
