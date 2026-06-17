"""Post-hoc audit for under-merged entity records.

The complement of ``merge_audit``: that module finds records where
*one* entry conflates multiple real tools. This one finds clusters
where *several* records describe the same real tool but the extraction
pipeline never recognised them as duplicates (e.g. ``UNSEEN APP`` /
``UNSEENUK MOBILE APP`` / ``UNSEEN / BT MODERN SLAVERY REPORTING APP``
— all the same product, but string-dissimilar enough that fuzzy +
arbiter at extraction time never paired them).

Pipeline:
  1. Cheap clustering — group records by Organization, then
     sub-cluster by overlap on label tokens, aliases, and description
     trigrams. Single-record "clusters" are dropped.
  2. Per cluster, ask an LLM whether some/all members refer to the
     same tool; if so, propose merge groups with a canonical label.
  3. Return ``DedupAuditResult`` per cluster; callers decide whether
     to surface, ignore, or apply the merges.

This module does NOT mutate the record set. Apply via
``BuildEntityDataset.apply_dedup_merge`` (the api wrapper).
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger("schemify.dedup_audit")


_PROMPT = """\
You are auditing a cluster of entity records that share an organisation
and look like they MIGHT describe the same underlying tool. They were
not merged automatically because their labels are too dissimilar for
string-based matching.

For each record you are given: its canonical label, aliases, the Tool
Description value(s) on the record, and other identifying attributes.

Decide which records describe the **same** real-world tool. Group them
into merge sets. A merge set is two or more records that should
collapse into one. Records that genuinely describe distinct tools (or
the parent organisation itself) should NOT appear in any merge set.

Be CONSERVATIVE. Only merge when the descriptions clearly describe the
same product (a mobile app and a website-reporting form for the same
helpline ARE the same service; a mobile app and a separate web portal
with a different name are NOT). When in doubt, do not merge.

For each merge set, pick a canonical label — prefer the shortest,
clearest, organisation-prefixed label among the members.

Output JSON with:
  - merge_sets: list of {{canonical_label, member_labels: [list of 2+
    labels from the cluster, MUST include canonical_label],
    rationale: short string}}.
  - confidence: 0.0-1.0 overall.
  - reason: one short sentence summarising the audit.

Organization: {organization}

Cluster ({n_records} records):
{records_block}
"""


_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "dedup_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "merge_sets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical_label": {"type": "string"},
                            "member_labels": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "canonical_label", "member_labels", "rationale",
                        ],
                        "additionalProperties": False,
                    },
                },
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["merge_sets", "confidence", "reason"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class MergeSet:
    canonical_label: str
    member_labels: list[str]
    rationale: str


@dataclass
class DedupAuditResult:
    organization: str
    member_labels: list[str] = field(default_factory=list)
    candidate_reason: str = ""
    confidence: float = 0.0
    reason: str = ""
    merge_sets: list[MergeSet] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "organization": self.organization,
            "member_labels": self.member_labels,
            "candidate_reason": self.candidate_reason,
            "confidence": self.confidence,
            "reason": self.reason,
            "merge_sets": [
                {
                    "canonical_label": m.canonical_label,
                    "member_labels": m.member_labels,
                    "rationale": m.rationale,
                }
                for m in self.merge_sets
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
    vals = _attr_values(rec, name)
    return vals[0] if vals else ""


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_GENERIC_TOKENS = frozenset({
    "the", "and", "of", "for", "to", "a", "an", "in", "on", "with",
    "by", "uk", "us", "usa", "international", "intl", "global",
    "tool", "tools", "system", "platform", "service", "services",
    "app", "application", "portal", "website", "web", "online",
    "mobile", "channel", "channels", "report", "reporting",
})


def _tokens(text: str) -> set[str]:
    return {
        t.lower() for t in _TOKEN_RE.findall(text or "")
        if len(t) >= 2 and t.lower() not in _GENERIC_TOKENS
    }


def _org_key(rec) -> str:
    org = (
        _attr_value(rec, "Organization Name")
        or _attr_value(rec, "Organization")
    )
    return org.strip().lower() if org else ""


def _signature_tokens(rec) -> set[str]:
    """Concatenate label + aliases + first description into a token set."""
    parts: list[str] = [getattr(rec, "label", "") or ""]
    parts.extend(getattr(rec, "aliases", None) or [])
    descs = (
        _attr_values(rec, "Tool Description")
        or _attr_values(rec, "Description")
    )
    if descs:
        parts.append(descs[0][:400])
    return _tokens(" ".join(parts))


def find_clusters(
    record_set, *, min_token_overlap: int = 2, max_cluster: int = 12
) -> list[tuple[str, list]]:
    """Cheap heuristic clustering. Returns list of (org, [records]).

    Records without an Organization, and singleton clusters, are
    dropped. Clusters larger than ``max_cluster`` are split into the
    densest sub-groups so the per-cluster LLM payload stays bounded.
    """
    if record_set is None or not getattr(record_set, "records", None):
        return []
    by_org: dict[str, list] = defaultdict(list)
    for r in record_set.records:
        ok = _org_key(r)
        if ok:
            by_org[ok].append(r)
    clusters: list[tuple[str, list]] = []
    for org, recs in by_org.items():
        if len(recs) < 2:
            continue
        # Union-find on token overlap.
        parent = list(range(len(recs)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        sigs = [_signature_tokens(r) for r in recs]
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if len(sigs[i] & sigs[j]) >= min_token_overlap:
                    union(i, j)
        groups: dict[int, list] = defaultdict(list)
        for i, r in enumerate(recs):
            groups[find(i)].append(r)
        # Read org display name from first record.
        org_display = (
            _attr_value(recs[0], "Organization Name")
            or _attr_value(recs[0], "Organization")
            or org
        )
        for group in groups.values():
            if len(group) < 2:
                continue
            # Split oversize clusters into chunks of max_cluster.
            for chunk_start in range(0, len(group), max_cluster):
                chunk = group[chunk_start:chunk_start + max_cluster]
                if len(chunk) >= 2:
                    clusters.append((org_display, chunk))
    return clusters


def _format_record(rec, idx: int, max_desc_chars: int) -> str:
    label = getattr(rec, "label", "") or ""
    aliases = list(getattr(rec, "aliases", None) or [])
    descs = (
        _attr_values(rec, "Tool Description")
        or _attr_values(rec, "Description")
    )
    desc = (descs[0] if descs else "(no description)")[:max_desc_chars]
    tech = _attr_value(rec, "Technology Type") or "(unknown)"
    func = _attr_value(rec, "Function") or _attr_value(rec, "Functionality") or "(unknown)"
    lines = [
        f"  [{idx}] label: {label}",
        f"      aliases: {', '.join(aliases) if aliases else '(none)'}",
        f"      tech_type: {tech}",
        f"      function: {func}",
        f"      description: {desc}",
    ]
    return "\n".join(lines)


async def _audit_one(
    llm,
    organization: str,
    cluster: list,
    sem: asyncio.Semaphore,
    max_desc_chars: int,
) -> DedupAuditResult:
    member_labels = [getattr(r, "label", "") or "" for r in cluster]
    records_block = "\n".join(
        _format_record(r, i, max_desc_chars) for i, r in enumerate(cluster)
    )
    async with sem:
        try:
            result = await llm.structured_completion(
                prompt=_PROMPT,
                response_format=_SCHEMA,
                variables={
                    "organization": organization,
                    "n_records": len(cluster),
                    "records_block": records_block,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return DedupAuditResult(
                organization=organization,
                member_labels=member_labels,
                candidate_reason=f"{len(cluster)} sibling records under {organization}",
                error=str(exc),
            )
    label_set = {lab.upper() for lab in member_labels}
    merge_sets: list[MergeSet] = []
    for ms in (result.get("merge_sets") or []):
        if not isinstance(ms, dict):
            continue
        canonical = str(ms.get("canonical_label") or "").strip()
        members = [str(m).strip() for m in (ms.get("member_labels") or []) if str(m).strip()]
        members = [m for m in members if m.upper() in label_set]
        if canonical.upper() not in {m.upper() for m in members}:
            # LLM proposed a canonical label that isn't a cluster member;
            # fall back to the first member.
            if not members:
                continue
            canonical = members[0]
        if len(members) < 2:
            continue
        merge_sets.append(MergeSet(
            canonical_label=canonical,
            member_labels=members,
            rationale=str(ms.get("rationale") or "").strip(),
        ))
    return DedupAuditResult(
        organization=organization,
        member_labels=member_labels,
        candidate_reason=f"{len(cluster)} sibling records under {organization}",
        confidence=float(result.get("confidence") or 0.0),
        reason=str(result.get("reason") or ""),
        merge_sets=merge_sets,
    )


async def audit_clusters(
    record_set,
    llm,
    *,
    concurrency: int = 8,
    max_desc_chars: int = 400,
    min_token_overlap: int = 2,
    max_cluster: int = 12,
    progress_cb=None,
) -> list[DedupAuditResult]:
    """Audit every dedup-candidate cluster in ``record_set``.

    Returns one ``DedupAuditResult`` per cluster. Clusters with no
    proposed merges still appear in the output (with ``merge_sets``
    empty) so callers can see what was considered.
    """
    if llm is None:
        return []
    clusters = find_clusters(
        record_set,
        min_token_overlap=min_token_overlap,
        max_cluster=max_cluster,
    )
    if not clusters:
        return []
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    tasks = [
        asyncio.create_task(
            _audit_one(llm, org, group, sem, max_desc_chars)
        )
        for org, group in clusters
    ]
    results: list[DedupAuditResult] = []
    for i, t in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await t)
        if progress_cb is not None:
            try:
                progress_cb(i, len(tasks))
            except Exception:  # noqa: BLE001
                pass
    return results


def flagged_results(
    results: Iterable[DedupAuditResult], *, confidence_threshold: float = 0.7
) -> list[DedupAuditResult]:
    """Filter results down to those with at least one proposed merge."""
    return [
        r for r in results
        if r.error is None
        and r.merge_sets
        and r.confidence >= confidence_threshold
    ]
