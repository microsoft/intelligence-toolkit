"""
Per-record value consolidation for attributes that accumulated values across
research rounds. (``value_cleanup`` tidies the *set* of categorical values;
this module tidies the values *within each record*.)

- ``select_single_values`` asks the LLM which of a record's several values best
  fits the named entity, e.g. one Tool Description instead of five near-copies,
  some describing a different tool. Nothing changes until
  ``apply_single_value_choices``.
- ``value_variant_changes`` finds spelling variants of one name within a record
  (``"Acme"`` / ``"Acme, Inc."`` / ``"Acme (ACM)"``) and combined values whose
  parts are also listed separately. Deterministic and free; nothing changes until
  ``apply_value_variant_changes``.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .models import RecordSet, SourcedValue
from .scope_audit import _classify_in_batches

PROMPT = """You are cleaning a dataset of "{category}".
Each entity below has several candidate values for the attribute "{attribute}"
({attribute_description}). Only one value should be kept.

For each entity choose the single candidate that most accurately describes the
named entity itself. Prefer a complete, specific sentence over a short fragment
or a title-cased heading. Never choose a candidate that describes a different
product, organization or entity than the one named, even if it is longer.

Entities (candidates are numbered from 0):
{entities}
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "single_value_choice",
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
                            "choice": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["label", "choice", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

_MAX_CANDIDATE_CHARS = 400


def _norm_text(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").casefold()).strip()


def _merge_sources(target: SourcedValue, others: list[SourcedValue]) -> None:
    seen = {(c.url, c.snippet) for c in target.sources}
    for other in others:
        for c in other.sources:
            if (c.url, c.snippet) not in seen:
                seen.add((c.url, c.snippet))
                target.sources.append(c)


async def select_single_values(
    rs: RecordSet,
    llm,
    attribute: str,
    *,
    context_attributes: Optional[list[str]] = None,
    batch_size: int = 8,
    concurrency: int = 6,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Return ``{"label", "choice", "chosen", "dropped", "reason"}`` for every record
    holding more than one value for ``attribute``. ``choice`` is ``None`` when the
    model gave no usable answer; applying then leaves that record unchanged."""
    schema = next((a for a in rs.schema_attributes if a.name == attribute), None)
    briefs, candidates = [], {}
    for record in rs.records:
        av = record.attributes.get(attribute)
        if not av or len(av.values) < 2:
            continue
        values = [sv.value for sv in av.values]
        candidates[record.label] = values
        brief = {"label": record.label, "aliases": list(record.aliases or [])[:5],
                 "candidates": [v[:_MAX_CANDIDATE_CHARS] for v in values]}
        for name in context_attributes or []:
            ctx = record.attributes.get(name)
            if ctx and ctx.values:
                brief[name] = "; ".join(sv.value for sv in ctx.values)[:200]
        briefs.append(brief)
    if not briefs:
        return []

    def fallback(label: str, reason: str) -> dict:
        return {"label": label, "choice": None, "chosen": None, "dropped": [], "reason": reason}

    def parse(brief: dict, item: dict) -> dict:
        values = candidates[brief["label"]]
        choice = item.get("choice")
        if not isinstance(choice, int) or not 0 <= choice < len(values):
            return fallback(brief["label"], f"invalid choice {choice!r}")
        return {"label": brief["label"], "choice": choice, "chosen": values[choice],
                "dropped": [v for i, v in enumerate(values) if i != choice],
                "reason": item.get("reason", "")}

    return await _classify_in_batches(
        llm, briefs,
        prompt=PROMPT,
        response_format=RESPONSE_FORMAT,
        variables={"category": rs.category, "attribute": attribute,
                   "attribute_description": (schema.description if schema else "") or "no description"},
        parse=parse, fallback=fallback,
        batch_size=batch_size, concurrency=concurrency, progress_cb=progress_cb,
    )


def apply_single_value_choices(rs: RecordSet, attribute: str, choices: list[dict]) -> int:
    """Keep only the chosen value per record and mark the attribute single-valued.

    Sources of dropped candidates with the same text (ignoring case/punctuation)
    fold into the kept value; candidates with different text are dropped together
    with their sources. Returns the number of records changed."""
    by_label = {r.label: r for r in rs.records}
    changed = 0
    for c in choices:
        record = by_label.get(c.get("label"))
        if record is None or c.get("choice") is None:
            continue
        av = record.attributes.get(attribute)
        if not av or len(av.values) < 2:
            continue
        kept = next((sv for sv in av.values if sv.value == c["chosen"]), None)
        if kept is None:
            continue
        same = [sv for sv in av.values if sv is not kept and _norm_text(sv.value) == _norm_text(kept.value)]
        _merge_sources(kept, same)
        av.values = [kept]
        changed += 1
    for a in rs.schema_attributes:
        if a.name == attribute:
            a.is_multi_valued = False
    return changed


_LEGAL_SUFFIX = re.compile(
    r"[\s,]+(inc|incorporated|ltd|limited|llc|llp|plc|gmbh|corp|corporation|co|pbc|pty)\.?$",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"\s*(?:;|,|/|\band\b|&)\s*", re.IGNORECASE)


def variant_key(value: str) -> str:
    """Key under which spelling variants of one name collapse: parentheticals,
    trailing legal suffixes, case and punctuation are ignored."""
    s = re.sub(r"\([^)]*\)", " ", value or "").strip()
    prev = None
    while prev != s:
        prev, s = s, _LEGAL_SUFFIX.sub("", s).strip()
    return re.sub(r"\W+", "", s.casefold())


def _variant_groups(values: list[str]) -> list[dict]:
    """Group values by ``variant_key``. A group is ``{"keep", "members", "key"}``;
    combined values whose parts are all listed on their own get ``keep=None`` and
    ``into=[part keys]``."""
    groups: dict[str, list[str]] = {}
    for v in values:
        groups.setdefault(variant_key(v), []).append(v)
    keys = set(groups)
    out = []
    for key, members in groups.items():
        parts = [variant_key(p) for p in _SPLIT.split(members[0]) if p.strip()]
        parts = [p for p in parts if p]
        if len(parts) > 1 and all(p in keys and p != key for p in parts):
            out.append({"keep": None, "members": members, "key": key, "into": parts})
        else:
            # Fullest form wins, e.g. "National Center for Missing & Exploited Children (NCMEC)".
            out.append({"keep": max(members, key=len), "members": members, "key": key})
    return out


def value_variant_changes(rs: RecordSet, attribute: str) -> list[dict]:
    """Preview: ``{"label", "before", "after"}`` for each record that would change."""
    changes = []
    for record in rs.records:
        av = record.attributes.get(attribute)
        if not av or len(av.values) < 2:
            continue
        before = [sv.value for sv in av.values]
        after = [g["keep"] for g in _variant_groups(before) if g["keep"] is not None]
        if after != before:
            changes.append({"label": record.label, "before": before, "after": after})
    return changes


def apply_value_variant_changes(rs: RecordSet, attribute: str) -> int:
    """Collapse spelling variants within each record, keeping the fullest form and all
    sources; drop combined values whose parts are listed separately, moving their
    sources to the parts. Returns the number of records changed."""
    changed = 0
    for record in rs.records:
        av = record.attributes.get(attribute)
        if not av or len(av.values) < 2:
            continue
        by_value: dict[str, SourcedValue] = {}
        for sv in av.values:
            if sv.value in by_value:
                _merge_sources(by_value[sv.value], [sv])
            else:
                by_value[sv.value] = sv
        groups = _variant_groups(list(by_value))
        kept: dict[str, SourcedValue] = {}
        for g in groups:
            if g["keep"] is not None:
                keep = by_value[g["keep"]]
                _merge_sources(keep, [by_value[m] for m in g["members"] if m != g["keep"]])
                kept[g["key"]] = keep
        for g in groups:
            if g["keep"] is None:
                for part in g["into"]:
                    _merge_sources(kept[part], [by_value[m] for m in g["members"]])
        new_values = list(kept.values())
        if [sv.value for sv in new_values] != [sv.value for sv in av.values]:
            av.values = new_values
            changed += 1
    return changed
