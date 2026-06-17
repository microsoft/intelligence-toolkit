"""LLM-driven cleanup of compound categorical values.

Runs after `Schemify.normalize()` to resolve compound values like
"Nonprofit / NGO" or "Maritime / Ports" that survived the fuzzy-clustering
step. The LLM decides per case: pick one synonym, split into independent
parts, or simplify a verbose label.

Why this exists: `auto_normalize()` clusters near-duplicates but does not
deduplicate semantic compounds. Without this pass, a closed-set attribute
ends up with redundant "A", "B", and "A / B" entries — breaking dashboard
filters and analysis.

Free in the common case: no LLM call is made when an attribute has no
compound values (no '/' or ';' in any value).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Callable, Optional

from .llm import LLMClient
from .models import RecordSet, SchemaAttribute

logger = logging.getLogger("schemify.cleanup")

BATCH_SIZE = 60

_PROMPT = """You are deduplicating and tidying categorical values for the
attribute "{attr_name}". The values were produced by messy LLM normalization
and contain redundant compounds.

CRITICAL: ANY value containing "/" or ";" is a compound and MUST be resolved
to one or more atomic values. NEVER return a compound unchanged.

For EACH value, return a list of cleaned atomic values:

  ATOMIC (no '/' or ';'): usually KEEP as-is. Only SIMPLIFY if verbose
                          (>40 chars or redundant trailing phrases).
                          → return [value] or [shorter_form]

  COMPOUND ('A / B' or 'A; B; C' etc.): resolve to atomic parts.
    - If parts are SYNONYMS / acronym+expansion (e.g. "Nonprofit / NGO",
      "CSAM / Child Sexual Exploitation"): pick the SINGLE most canonical
      atomic form. Prefer the shortest unambiguous label.
      → return [chosen_one]
    - If parts are INDEPENDENT concepts (e.g. "Labor / Sexual Abuse",
      "Forced Marriage / Honour-Based Abuse"): split into all parts.
      → return [partA, partB, ...]
    - NEVER return the compound string itself.

When an atomic value already appears elsewhere in the input list, prefer
to map compounds to that existing label so the vocabulary stays consistent.

Examples (illustrative):
  "Nonprofit / NGO"                              → ["Nonprofit"]
  "CSAM / Child Sexual Exploitation and Abuse"   → ["CSAM"]
  "Maritime / Ports"                             → ["Maritime", "Ports"]
  "Forced Marriage / Honour-Based Abuse"         → ["Forced Marriage", "Honour-Based Abuse"]
  "Labor; Forced Labor; Child Labor"             → ["Labor", "Forced Labor", "Child Labor"]
  "Human Trafficking and Exploitation Prevention / Support" → ["Trafficking Prevention"]

Return JSON: {{"results": [{{"original": "<input>", "cleaned": ["<part1>", ...]}}, ...]}}
Every input value must appear once as "original". Cleaned values must be
atomic (no '/' or ';'), concise, and consistent across the batch.

Values to clean:
{values}
"""

_SCHEMA = {
    "json_schema": {
        "name": "value_cleanup",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "cleaned": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["original", "cleaned"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    }
}


def _has_compound(val: str) -> bool:
    return "/" in val or ";" in val


def _collect_distinct(record_set: RecordSet, attr_name: str) -> set[str]:
    out: set[str] = set()
    for rec in record_set.records:
        attr = rec.attributes.get(attr_name)
        if attr is None:
            continue
        for sv in (attr.values or []):
            v = getattr(sv, "value", None)
            if v:
                out.add(str(v))
    return out


def _mechanical_split(val: str) -> list[str]:
    parts = [
        p.strip()
        for p in val.replace(" / ", "/").replace(";", "/").split("/")
        if p.strip()
    ]
    return parts or [val]


async def _clean_batch(
    llm: LLMClient, attr_name: str, values: list[str]
) -> dict[str, list[str]]:
    values_json = json.dumps(values, ensure_ascii=False, indent=2)
    try:
        result = await llm.structured_completion(
            prompt=_PROMPT,
            response_format=_SCHEMA,
            variables={"attr_name": attr_name, "values": values_json},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("compound-cleanup batch failed for %s: %s", attr_name, e)
        # Fallback: mechanically split any compounds, keep atoms as-is
        return {v: (_mechanical_split(v) if _has_compound(v) else [v]) for v in values}

    items = result.get("results", []) if isinstance(result, dict) else []
    mapping: dict[str, list[str]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        k = it.get("original")
        cleaned = it.get("cleaned") or []
        if not (isinstance(k, str) and isinstance(cleaned, list) and cleaned):
            continue
        flat: list[str] = []
        for c in cleaned:
            if not c:
                continue
            cs = str(c)
            # Safety net: LLM occasionally returns compounds anyway
            if _has_compound(cs):
                flat.extend(_mechanical_split(cs))
            else:
                flat.append(cs)
        if flat:
            mapping[k] = flat

    # Defensive: ensure every input mapped — never leave a compound unresolved
    for v in values:
        if v not in mapping or not mapping[v]:
            mapping[v] = _mechanical_split(v) if _has_compound(v) else [v]
    return mapping


def _apply_to_record_set(
    record_set: RecordSet, attr_name: str, mapping: dict[str, list[str]]
) -> int:
    """Rewrite attr.values entries in place. Returns count of records changed."""
    changed = 0
    for rec in record_set.records:
        attr = rec.attributes.get(attr_name)
        if attr is None or not attr.values:
            continue

        new_values: list = []
        # Merge SourcedValues by cleaned value, accumulating sources
        merged: dict[str, Any] = {}
        for sv in attr.values:
            orig = str(getattr(sv, "value", "") or "")
            if not orig:
                continue
            for cleaned in mapping.get(orig, [orig]):
                if not cleaned:
                    continue
                if cleaned not in merged:
                    merged[cleaned] = replace(
                        sv, value=cleaned, sources=list(sv.sources or [])
                    )
                else:
                    existing = merged[cleaned]
                    seen_urls = {
                        getattr(s, "url", None)
                        for s in (existing.sources or [])
                    }
                    for s in (sv.sources or []):
                        if getattr(s, "url", None) not in seen_urls:
                            existing.sources.append(s)
                            seen_urls.add(getattr(s, "url", None))

        new_values = list(merged.values())
        # Only update if changed
        if [getattr(v, "value", None) for v in new_values] != [
            getattr(v, "value", None) for v in attr.values
        ]:
            attr.values = new_values
            changed += 1
    return changed


async def clean_compound_values(
    record_set: RecordSet,
    schema_attributes: list[SchemaAttribute],
    llm: LLMClient,
    attributes: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, dict]:
    """Resolve compound values (e.g. 'A / B', 'A; B') via the LLM.

    Operates on closed-set schema attributes by default. No-op for attributes
    that have no compound values (zero LLM cost in the clean case).

    Returns a summary dict per attribute: {n_values, n_compounds, n_changed}.
    """
    targets = [
        a for a in schema_attributes
        if a.is_closed_set and (attributes is None or a.name in attributes)
    ]
    total = len(targets)
    summary: dict[str, dict] = {}

    for idx, attr in enumerate(targets):
        if progress_callback:
            try:
                progress_callback(idx, total, attr.name)
            except Exception:  # noqa: BLE001
                pass

        distinct = sorted(_collect_distinct(record_set, attr.name))
        compounds = [v for v in distinct if _has_compound(v)]
        if not compounds:
            summary[attr.name] = {
                "n_values": len(distinct),
                "n_compounds": 0,
                "n_changed": 0,
            }
            continue

        logger.info(
            "compound cleanup: %s — %d distinct, %d compound",
            attr.name, len(distinct), len(compounds),
        )

        mapping: dict[str, list[str]] = {}
        for i in range(0, len(distinct), BATCH_SIZE):
            batch = distinct[i : i + BATCH_SIZE]
            batch_map = await _clean_batch(llm, attr.name, batch)
            mapping.update(batch_map)

        changed = _apply_to_record_set(record_set, attr.name, mapping)
        summary[attr.name] = {
            "n_values": len(distinct),
            "n_compounds": len(compounds),
            "n_changed": changed,
        }

    if progress_callback:
        try:
            progress_callback(total, total, "compound-cleanup complete")
        except Exception:  # noqa: BLE001
            pass

    return summary
