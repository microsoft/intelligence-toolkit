"""Post-hoc audit for over-merged entity records.

A bad merge typically looks like:
  - One record carrying multiple Tool Description / Description values
    that describe materially different functionality (a mobile app + a
    server-side remediation system).
  - Aliases that look like sibling products from a vendor catalog page
    (sharing a "PREFIX: X" / "PREFIX: Y" template with the canonical
    label).

This module:
  1. Heuristically picks candidate records (multi-description or
     vendor-catalog aliases).
  2. Asks an LLM, per candidate, whether the record is one entity or
     several; if several, asks for a split proposal.
  3. Returns a structured AuditResult per candidate; callers decide
     whether to surface, ignore, or apply the splits.

This module does NOT mutate the record set. See
``BuildEntityDataset.apply_audit_splits`` (api wrapper) for the
mutation side.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger("schemify.merge_audit")


_PROMPT = """\
You are auditing one entity record from a curated tools dataset. The
record may legitimately describe a single tool — or it may be a bad
merge that conflated two or more sibling products from the same vendor's
catalog page (e.g. a vendor lists "App X" and "System Y" together, and
the extractor mashed both onto one record).

You are given:
  - The record's canonical label and aliases.
  - All Tool Description values currently on the record (often each was
    extracted from a different source URL).
  - Other identifying attributes (organization, tech type, function).

Decide:
  - is_single_entity: true if every description plausibly refers to the
    same real-world tool. false if descriptions clearly describe
    different products (e.g. a mobile app vs. a server-side system,
    a dataset vs. an analysis tool, a hotline number vs. a web portal).
  - confidence: 0.0-1.0 in your decision.
  - reason: one short sentence.
  - split_proposal: if is_single_entity is false, return a list of
    proposed sub-entities; each item: {{label, description_indices,
    rationale}} where description_indices is a list of 0-based indices
    into the descriptions array indicating which descriptions belong
    to that sub-entity. If is_single_entity is true, return [].

Be CONSERVATIVE. If descriptions just paraphrase each other or describe
features of the same product, return is_single_entity=true. Only flag
clear cases where the record contains materially different products.

Record:
  label: {label}
  aliases: {aliases}
  organization: {organization}
  tech_type: {tech_type}
  function: {function}

Descriptions ({n_descriptions}):
{descriptions_block}
"""


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "merge_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_single_entity": {"type": "boolean"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
                "split_proposal": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "description_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["label", "description_indices", "rationale"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "is_single_entity", "confidence", "reason", "split_proposal",
            ],
            "additionalProperties": False,
        },
    },
}


@dataclass
class SplitProposal:
    label: str
    description_indices: list[int]
    rationale: str


@dataclass
class AuditResult:
    label: str
    aliases: list[str] = field(default_factory=list)
    is_single_entity: bool = True
    confidence: float = 0.0
    reason: str = ""
    candidate_reason: str = ""
    n_descriptions: int = 0
    descriptions: list[str] = field(default_factory=list)
    split_proposal: list[SplitProposal] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "aliases": self.aliases,
            "is_single_entity": self.is_single_entity,
            "confidence": self.confidence,
            "reason": self.reason,
            "candidate_reason": self.candidate_reason,
            "n_descriptions": self.n_descriptions,
            "descriptions": self.descriptions,
            "split_proposal": [
                {
                    "label": s.label,
                    "description_indices": s.description_indices,
                    "rationale": s.rationale,
                }
                for s in self.split_proposal
            ],
            "error": self.error,
        }


def _attr_values(rec, name: str) -> list[str]:
    for bucket in ("attributes", "additional_attributes"):
        b = getattr(rec, bucket, None) or {}
        a = b.get(name)
        if a is None:
            continue
        vals: list[str] = []
        for sv in (getattr(a, "values", None) or []):
            v = getattr(sv, "value", None)
            if isinstance(v, str) and v.strip():
                vals.append(v.strip())
        if vals:
            return vals
    return []


def _attr_value(rec, name: str) -> str:
    for bucket in ("attributes", "additional_attributes"):
        b = getattr(rec, bucket, None) or {}
        a = b.get(name)
        if a is None:
            continue
        v = getattr(a, "value", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            return ", ".join(str(x) for x in v if x)
    return ""


def is_candidate(rec) -> tuple[bool, str]:
    """Cheap heuristic: should this record be sent to the LLM auditor?"""
    descs = _attr_values(rec, "Tool Description") or _attr_values(rec, "Description")
    if len(descs) >= 2:
        return True, f"{len(descs)} Tool Description values"
    aliases = list(getattr(rec, "aliases", None) or [])
    label = getattr(rec, "label", "") or ""
    for a in aliases:
        if ":" in a and ":" in label:
            a_prefix = a.split(":", 1)[0].strip().upper()
            l_prefix = label.split(":", 1)[0].strip().upper()
            if a_prefix and a_prefix == l_prefix:
                return True, f"shared catalog prefix '{a_prefix}:' with alias"
    return False, ""


async def _audit_one(
    llm,
    rec,
    candidate_reason: str,
    sem: asyncio.Semaphore,
    max_desc_chars: int,
    max_descs: int,
) -> AuditResult:
    descs = (
        _attr_values(rec, "Tool Description")
        or _attr_values(rec, "Description")
    )
    if not descs:
        descs = ["(no description on record)"]
    seen: set[str] = set()
    unique: list[str] = []
    for d in descs:
        k = " ".join(d.split()).lower()[:160]
        if k in seen:
            continue
        seen.add(k)
        unique.append(d)
    trimmed = [d[:max_desc_chars] for d in unique[:max_descs]]
    desc_block = "\n".join(f"  [{i}] {d}" for i, d in enumerate(trimmed))
    label = getattr(rec, "label", "") or ""
    aliases = list(getattr(rec, "aliases", None) or [])
    async with sem:
        try:
            result = await llm.structured_completion(
                prompt=_PROMPT,
                response_format=_SCHEMA,
                variables={
                    "label": label,
                    "aliases": ", ".join(aliases) or "(none)",
                    "organization": (
                        _attr_value(rec, "Organization Name")
                        or _attr_value(rec, "Organization") or "(unknown)"
                    ),
                    "tech_type": _attr_value(rec, "Technology Type") or "(unknown)",
                    "function": _attr_value(rec, "Function") or "(unknown)",
                    "n_descriptions": len(unique),
                    "descriptions_block": desc_block,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return AuditResult(
                label=label,
                aliases=aliases,
                candidate_reason=candidate_reason,
                n_descriptions=len(unique),
                descriptions=trimmed,
                error=str(exc),
            )
    splits = []
    for s in (result.get("split_proposal") or []):
        if not isinstance(s, dict):
            continue
        try:
            splits.append(SplitProposal(
                label=str(s.get("label") or "").strip(),
                description_indices=[int(i) for i in (s.get("description_indices") or [])],
                rationale=str(s.get("rationale") or "").strip(),
            ))
        except Exception:  # noqa: BLE001
            continue
    return AuditResult(
        label=label,
        aliases=aliases,
        is_single_entity=bool(result.get("is_single_entity", True)),
        confidence=float(result.get("confidence") or 0.0),
        reason=str(result.get("reason") or ""),
        candidate_reason=candidate_reason,
        n_descriptions=len(unique),
        descriptions=trimmed,
        split_proposal=splits,
    )


async def audit_records(
    record_set,
    llm,
    *,
    concurrency: int = 8,
    max_desc_chars: int = 400,
    max_descs_per_record: int = 20,
    progress_cb=None,
) -> list[AuditResult]:
    """Audit every candidate record in ``record_set``. Returns a list of
    :class:`AuditResult` — one per candidate (records that don't trip
    the cheap heuristic are skipped, not returned).
    """
    if record_set is None or not getattr(record_set, "records", None):
        return []
    if llm is None:
        return []
    candidates: list[tuple[object, str]] = []
    for r in record_set.records:
        ok, why = is_candidate(r)
        if ok:
            candidates.append((r, why))
    if not candidates:
        return []
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    tasks = [
        asyncio.create_task(
            _audit_one(llm, r, why, sem, max_desc_chars, max_descs_per_record)
        )
        for r, why in candidates
    ]
    results: list[AuditResult] = []
    for i, t in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await t)
        if progress_cb is not None:
            try:
                progress_cb(i, len(tasks))
            except Exception:  # noqa: BLE001
                pass
    return results


def flagged_results(
    results: Iterable[AuditResult], *, confidence_threshold: float = 0.7
) -> list[AuditResult]:
    """Filter results down to those the auditor recommends splitting."""
    return [
        r for r in results
        if r.error is None
        and not r.is_single_entity
        and r.confidence >= confidence_threshold
        and r.split_proposal
    ]


def _sv_urls(sv) -> set[str]:
    urls: set[str] = set()
    for s in (getattr(sv, "sources", None) or []):
        url = getattr(s, "url", None)
        if isinstance(url, str) and url:
            urls.add(url)
    return urls


def _desc_pool(rec) -> list[tuple[str, object, str, str]]:
    """Return [(value, sv, attr_name, bucket_name)] for description SVs."""
    out: list[tuple[str, object, str, str]] = []
    for attr_name in ("Tool Description", "Description"):
        for bucket in ("attributes", "additional_attributes"):
            a = (getattr(rec, bucket, None) or {}).get(attr_name)
            if a is None:
                continue
            for sv in (getattr(a, "values", None) or []):
                v = getattr(sv, "value", None)
                if isinstance(v, str) and v.strip():
                    out.append((v, sv, attr_name, bucket))
            if out:
                return out
    return out


def _match_desc(
    target: str, pool: list[tuple[str, object, str, str]], used: set[int]
) -> Optional[int]:
    norm_t = " ".join(target.split()).lower()[:120]
    if not norm_t:
        return None
    for i, (val, _sv, _name, _bucket) in enumerate(pool):
        if i in used:
            continue
        norm_v = " ".join(val.split()).lower()[:120]
        if (
            norm_v == norm_t
            or (len(norm_t) >= 80 and norm_v.startswith(norm_t[:80]))
            or (len(norm_v) >= 80 and norm_t.startswith(norm_v[:80]))
        ):
            return i
    return None


def apply_split(
    rec, audit: AuditResult
) -> tuple[list, list[tuple[str, str]]]:
    """Apply an audit's split_proposal to ``rec``. Returns a tuple:
    ``(new_records, do_not_merge_pairs)``. If the split cannot be
    applied (no descriptions, no proposals) returns ``([rec], [])`` so
    the caller can leave the original in place.
    """
    import copy as _copy
    from .models import AttributeValue, Record  # local: avoid cycles

    proposals = audit.split_proposal or []
    if not proposals:
        return [rec], []
    desc_pool = _desc_pool(rec)
    if not desc_pool:
        return [rec], []

    audit_descs = audit.descriptions or []
    per_split_urls: list[set[str]] = []
    per_split_descs: list[list] = []
    used: set[int] = set()
    for split in proposals:
        urls: set[str] = set()
        owned: list = []
        for idx in (split.description_indices or []):
            if not isinstance(idx, int) or idx < 0 or idx >= len(audit_descs):
                continue
            pool_idx = _match_desc(audit_descs[idx], desc_pool, used)
            if pool_idx is None:
                continue
            used.add(pool_idx)
            _val, sv, _name, _bucket = desc_pool[pool_idx]
            urls |= _sv_urls(sv)
            owned.append(sv)
        per_split_urls.append(urls)
        per_split_descs.append(owned)

    # Orphan descriptions → first split (canonical)
    for i, (_val, sv, _name, _bucket) in enumerate(desc_pool):
        if i not in used:
            per_split_urls[0] |= _sv_urls(sv)
            per_split_descs[0].append(sv)

    desc_attr_name, desc_attr_bucket = desc_pool[0][2], desc_pool[0][3]
    new_records: list = []
    for split_i, split in enumerate(proposals):
        new = Record(
            label=(split.label or f"{rec.label} (split {split_i+1})").strip()
        )
        if per_split_descs[split_i]:
            desc_attr = AttributeValue(
                values=[_copy.deepcopy(sv) for sv in per_split_descs[split_i]]
            )
            target = getattr(new, desc_attr_bucket)
            target[desc_attr_name] = desc_attr
        new_records.append(new)

    # Distribute every other attribute by source-URL intersection.
    for bucket_name in ("attributes", "additional_attributes"):
        for attr_name, attr in (getattr(rec, bucket_name, None) or {}).items():
            if attr_name == desc_attr_name:
                continue
            for sv in (getattr(attr, "values", None) or []):
                sv_urls = _sv_urls(sv)
                if sv_urls:
                    targets = [
                        i for i, urls in enumerate(per_split_urls)
                        if urls & sv_urls
                    ]
                    if not targets:
                        targets = [0]
                else:
                    targets = [0]
                for ti in targets:
                    target_bucket = getattr(new_records[ti], bucket_name)
                    existing = target_bucket.get(attr_name)
                    if existing is None:
                        existing = AttributeValue(values=[])
                        target_bucket[attr_name] = existing
                    existing.values.append(_copy.deepcopy(sv))

    # Preserve the original label as an alias on the first split (so
    # downstream lookups by old name still resolve).
    if new_records and rec.label and rec.label not in new_records[0].aliases:
        new_records[0].aliases.append(rec.label)

    # Compute pairwise do-not-merge constraints for the siblings + the
    # original label so the next merge pass won't reunite them.
    do_not_merge: list[tuple[str, str]] = []
    labels = [r.label for r in new_records]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            do_not_merge.append((labels[i], labels[j]))
        if rec.label and rec.label not in labels:
            do_not_merge.append((rec.label, labels[i]))
    return new_records, do_not_merge
