"""
Scope audit: flag records that are not instances of the dataset's category.

Discovery searches surface adjacent things — reports, policies, campaigns,
events, organizations without a product, or data sources *about* the topic.
This pass classifies each record against ``record_set.category`` and returns
reviewable results; it never removes anything itself.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Optional

from .models import RecordSet

ENTITY_KINDS = [
    "Product / Tool / Platform",
    "Dataset / Database",
    "Organization",
    "Program / Initiative",
    "Campaign / Awareness Effort",
    "Report / Publication",
    "Policy / Law / Standard",
    "Event",
    "Person",
    "Other",
]

PROMPT = """You are auditing a dataset whose entities should all be: {category}

Dataset guidance:
{guidance}

For each entity below decide whether it is itself an instance of that category.
Flag (in_scope=false) only when the evidence clearly shows it is something else, e.g.
a report or publication, a law, policy, code of conduct or standard document, an
awareness campaign, an event, a person, an organization described without any
qualifying product or service, or a source that is merely ABOUT the topic or used
as data by others rather than an instance of the category.
When an organization or program ships a qualifying product, it is in scope.
A product's relevance to the topic is NOT a scope question: if it is a real product
of the right kind but its documented use for the topic is weak or general-purpose,
keep in_scope=true and set weak_relevance=true instead.
Default to in_scope=true when uncertain, with low confidence.

Use one of these entity_kind values: {kinds}

For each entity return: label (echo exactly), in_scope, weak_relevance, entity_kind,
confidence (0-1, how sure you are of the in_scope decision), reason (one short sentence).

Entities:
{entities}
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "scope_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "in_scope": {"type": "boolean"},
                            "weak_relevance": {"type": "boolean"},
                            "entity_kind": {"type": "string", "enum": ENTITY_KINDS},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["label", "in_scope", "weak_relevance", "entity_kind", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

_MAX_VALUE_CHARS = 240


def entity_brief(record, attr_names: list[str]) -> dict:
    """Compact, schema-agnostic summary of a record for classification."""
    brief: dict = {"label": record.label, "aliases": list(record.aliases or [])[:5]}
    for name in attr_names:
        av = record.attributes.get(name)
        if av and av.values:
            joined = "; ".join(sorted({sv.value for sv in av.values if sv.value}))
            brief[name] = joined[:_MAX_VALUE_CHARS]
    return brief


def _unclassified(label: str, reason: str) -> dict:
    # Errors keep the record: an audit failure must never look like a removal signal.
    return {"label": label, "in_scope": True, "weak_relevance": False, "entity_kind": "Other",
            "confidence": 0.0, "reason": reason}


async def audit_scope(
    record_set: RecordSet,
    llm,
    *,
    batch_size: int = 15,
    concurrency: int = 6,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Classify every record; returns one result dict per record."""
    attr_names = [a.name for a in record_set.schema_attributes]
    briefs = [entity_brief(r, attr_names) for r in record_set.records]
    batches = [briefs[i:i + batch_size] for i in range(0, len(briefs), batch_size)]
    sem = asyncio.Semaphore(max(1, concurrency))
    done = [0]

    async def run(batch: list[dict]) -> list[dict]:
        async with sem:
            try:
                result = await llm.structured_completion(
                    prompt=PROMPT,
                    response_format=RESPONSE_FORMAT,
                    variables={
                        "category": record_set.category,
                        "guidance": record_set.guidance or "(none)",
                        "kinds": ", ".join(ENTITY_KINDS),
                        "entities": json.dumps(batch, ensure_ascii=False, indent=1),
                    },
                )
                by_label = {
                    it.get("label"): it for it in (result.get("results") or [])
                    if isinstance(it, dict)
                }
            except Exception as e:  # noqa: BLE001
                by_label = {}
                err = f"audit error: {e}"
            else:
                err = "missing from model response"
            out = []
            for b in batch:
                it = by_label.get(b["label"])
                if it is None:
                    out.append(_unclassified(b["label"], err))
                else:
                    out.append({
                        "label": b["label"],
                        "in_scope": bool(it.get("in_scope", True)),
                        "weak_relevance": bool(it.get("weak_relevance", False)),
                        "entity_kind": it.get("entity_kind") or "Other",
                        "confidence": float(it.get("confidence") or 0.0),
                        "reason": str(it.get("reason") or ""),
                    })
            done[0] += 1
            if progress_cb:
                progress_cb(done[0], len(batches))
            return out

    results = await asyncio.gather(*[run(b) for b in batches])
    return [r for batch in results for r in batch]


def flagged_results(results: list[dict], confidence_threshold: float = 0.7) -> list[dict]:
    return [
        r for r in results
        if not r.get("in_scope", True) and r.get("confidence", 0.0) >= confidence_threshold
    ]


def weak_relevance_results(results: list[dict]) -> list[dict]:
    """In-scope records whose documented relevance is weak: for human review, never auto-removal."""
    return [r for r in results if r.get("in_scope", True) and r.get("weak_relevance")]
