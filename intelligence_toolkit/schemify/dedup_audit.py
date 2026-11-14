"""Post-hoc audit for under-merged entity records and misattached aliases.

The complement of ``merge_audit``: that module finds records where
*one* entry conflates multiple real tools. This one finds groups where
*several* records describe the same real tool but the extraction
pipeline never recognised them as duplicates (e.g. ``UNSEEN APP`` /
``UNSEENUK MOBILE APP`` / ``MODERN SLAVERY & EXPLOITATION HELPLINE APP``),
and aliases that name a different entity than the record they sit on
(e.g. ``STOP APP`` attached to another organisation's platform).

Pipeline:
  1. Cheap candidate generation, without an LLM:
     - records whose operating organisation matches after normalisation
       (legal suffixes, country tags and punctuation removed), sub-grouped
       by overlap on label, alias and description tokens; the
       organisation attribute is detected from the schema, so
       ``Operating Organization``, ``Developer`` etc. all work;
     - records that share a name, i.e. one record's label or alias equals
       another's after normalisation, regardless of organisation.
  2. Per candidate group, ask an LLM which records describe the same
     entity (merge sets) and which aliases belong to a different entity
     (alias corrections).
  3. Return ``DedupAuditResult`` per group; callers decide whether to
     surface, ignore, or apply the proposals.

The audit never mutates the record set. ``apply_merge_set``,
``apply_alias_correction`` and ``apply_results`` apply reviewed
proposals and log each change in ``record_set.history``.
"""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

logger = logging.getLogger("schemify.dedup_audit")


_PROMPT = """\
You are auditing a small group of records from a dataset of: {category}.
The records were grouped for review because {reason}.

For each record you are given its label, aliases, operating
organisation(s), technology type, function, description(s) and a few
source URLs. Judge only from this information: do not fetch URLs or use
any tools.

Task 1 — merge_sets. Decide which records describe the **same**
real-world entity under different names. A merge set is two or more
records that should collapse into one. Be CONSERVATIVE:
- Merge spelling, prefix and naming variants of the same product or
  service (e.g. "STOP APP" and "THE STOP APP"; "UNSEEN APP" and
  "UNSEENUK MOBILE APP"; a helpline and the helpline's "digital contact
  system").
- Do NOT merge different products from the same organisation (e.g. a
  mobile app and a separate business portal; a dataset and an analytics
  platform; a research paper and a deployed tool). Sharing an
  organisation or a topic is not enough.
- Do NOT merge an umbrella programme or the organisation itself into a
  specific product unless the record plainly describes that product.
- When in doubt, do not merge.
For each merge set pick canonical_label: the clearest, commonly used
name of the entity, chosen from the member labels.

Task 2 — alias_corrections. Identify aliases that name a DIFFERENT
entity than the record they are listed on: another record's product,
another organisation's tool, the title of an unrelated study, or
placeholder text such as "None". For each, give the record's label, the
exact alias text, belongs_to_label (the label of the record in this
group that the alias actually names, or "" if none) and a short
rationale. Do not flag abbreviations, former names, translations or
spelling variants of the record's own entity. Ignore aliases of records
you are merging that will become aliases of the merged record anyway.

Output JSON with:
  - merge_sets: list of {{canonical_label, member_labels: [2+ labels
    from the group, MUST include canonical_label], rationale}}.
  - alias_corrections: list of {{record_label, alias, belongs_to_label,
    rationale}}.
  - confidence: 0.0-1.0 overall.
  - reason: one short sentence summarising the audit.

Group ({n_records} records):
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
                "alias_corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "record_label": {"type": "string"},
                            "alias": {"type": "string"},
                            "belongs_to_label": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "record_label", "alias", "belongs_to_label", "rationale",
                        ],
                        "additionalProperties": False,
                    },
                },
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["merge_sets", "alias_corrections", "confidence", "reason"],
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
class AliasCorrection:
    record_label: str
    alias: str
    belongs_to_label: str = ""
    rationale: str = ""


@dataclass
class CandidateCluster:
    organization: str
    records: list
    reason: str


@dataclass
class DedupAuditResult:
    organization: str
    member_labels: list[str] = field(default_factory=list)
    candidate_reason: str = ""
    confidence: float = 0.0
    reason: str = ""
    merge_sets: list[MergeSet] = field(default_factory=list)
    alias_corrections: list[AliasCorrection] = field(default_factory=list)
    error: str | None = None

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
            "alias_corrections": [
                {
                    "record_label": c.record_label,
                    "alias": c.alias,
                    "belongs_to_label": c.belongs_to_label,
                    "rationale": c.rationale,
                }
                for c in self.alias_corrections
            ],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DedupAuditResult:
        return cls(
            organization=data.get("organization", ""),
            member_labels=list(data.get("member_labels") or []),
            candidate_reason=data.get("candidate_reason", ""),
            confidence=float(data.get("confidence") or 0.0),
            reason=data.get("reason", ""),
            merge_sets=[
                MergeSet(
                    canonical_label=m.get("canonical_label", ""),
                    member_labels=list(m.get("member_labels") or []),
                    rationale=m.get("rationale", ""),
                )
                for m in data.get("merge_sets") or []
            ],
            alias_corrections=[
                AliasCorrection(
                    record_label=c.get("record_label", ""),
                    alias=c.get("alias", ""),
                    belongs_to_label=c.get("belongs_to_label", ""),
                    rationale=c.get("rationale", ""),
                )
                for c in data.get("alias_corrections") or []
            ],
            error=data.get("error"),
        )


# ── Attribute access ─────────────────────────────────────────────────


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


def _first_values(rec, names: Iterable[str]) -> list[str]:
    for name in names:
        vals = _attr_values(rec, name)
        if vals:
            return vals
    return []


_LEGACY_ORG_ATTRS = ("Organization Name", "Organization")
_ORG_ATTR_RE = re.compile(
    r"organi[sz]ation|operator|developer|vendor|provider|company|maker|publisher|owner",
    re.IGNORECASE,
)
_ORG_ATTR_EXCLUDE_RE = re.compile(
    r"\b(type|types|category|categories|size|kind|sector|count|country|location)\b",
    re.IGNORECASE,
)
_DESCRIPTION_ATTRS = ("Tool Description", "Description")
_FUNCTION_ATTRS = ("Functionality", "Function")


def org_attribute_names(record_set) -> list[str]:
    """Schema attributes that name the entity's operating organisation.

    Detected by name so datasets using ``Operating Organization``,
    ``Developer``, ``Vendor`` etc. are handled, falling back to the legacy
    ``Organization Name`` / ``Organization`` attributes.
    """
    names = [
        a.name for a in (getattr(record_set, "schema_attributes", None) or [])
        if _ORG_ATTR_RE.search(a.name or "") and not _ORG_ATTR_EXCLUDE_RE.search(a.name or "")
    ]
    for legacy in _LEGACY_ORG_ATTRS:
        if legacy not in names:
            names.append(legacy)
    return names


# ── Normalisation ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_GENERIC_TOKENS = frozenset({
    "the", "and", "of", "for", "to", "a", "an", "in", "on", "with",
    "by", "uk", "us", "usa", "international", "intl", "global",
    "tool", "tools", "system", "platform", "service", "services",
    "app", "application", "portal", "website", "web", "online",
    "mobile", "channel", "channels", "report", "reporting",
})
_ORG_SUFFIX_TOKENS = frozenset({
    "the", "inc", "incorporated", "ltd", "limited", "llc", "llp", "plc",
    "corp", "corporation", "co", "company", "group", "gmbh", "ag", "sa",
    "bv", "nv", "pty", "pte", "uk", "us", "usa",
})
_GENERIC_ORGS = frozenset({
    "academic research", "academic researchers", "researchers", "research",
    "university", "universities", "government", "governments", "various",
    "multiple", "unknown", "none", "n a", "na", "not specified", "other",
    "open source", "open source community", "community", "independent",
    "private sector", "civil society", "ngo", "ngos", "nonprofit",
})
_JUNK_NAMES = frozenset({"none", "null", "n a", "na", "nan", "unknown", "not applicable"})


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(_fold(text))
        if len(t) >= 2 and t not in _GENERIC_TOKENS
    }


def _fold(text: str) -> str:
    """Casefold and strip accents (``"Verité"`` → ``"verite"``)."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def normalize_name(name: str) -> str:
    """Case-, accent-, punctuation- and article-insensitive key for entity names."""
    text = re.sub(r"[^a-z0-9]+", " ", _fold(name)).strip()
    text = re.sub(r"^the ", "", text)
    return "" if text in _JUNK_NAMES else text


def normalize_org(name: str) -> set[str]:
    """Return comparable keys for one organisation value.

    ``"Issara Institute, Inc."`` → ``{"issara institute"}``;
    ``"Unseen UK"`` → ``{"unseen"}``; ``"A; B"`` → ``{"a", "b"}``.
    Parenthetical text and generic placeholders are dropped.
    """
    text = re.sub(r"\([^)]*\)?", " ", name or "")
    text = text.replace(".", "")
    keys: set[str] = set()
    for part in re.split(r"[;,|/]", text):
        toks = re.findall(r"[a-z0-9]+", _fold(part))
        while toks and toks[-1] in _ORG_SUFFIX_TOKENS:
            toks.pop()
        while toks and toks[0] == "the":
            toks.pop(0)
        key = " ".join(toks)
        if len(key) >= 3 and key not in _GENERIC_ORGS and key not in _ORG_SUFFIX_TOKENS:
            keys.add(key)
    return keys


def _record_org_values(rec, org_attrs: list[str]) -> list[str]:
    out: list[str] = []
    for name in org_attrs:
        for v in _attr_values(rec, name):
            if v not in out:
                out.append(v)
    return out


def _org_key(rec, org_attrs: list[str] | None = None) -> str:
    """Backward-compatible single organisation key (first value)."""
    vals = _record_org_values(rec, list(org_attrs or _LEGACY_ORG_ATTRS))
    for v in vals:
        keys = sorted(normalize_org(v))
        if keys:
            return keys[0]
    return ""


def _signature_tokens(rec) -> set[str]:
    """Concatenate label + aliases + first description into a token set."""
    parts: list[str] = [getattr(rec, "label", "") or ""]
    parts.extend(getattr(rec, "aliases", None) or [])
    descs = _first_values(rec, _DESCRIPTION_ATTRS)
    if descs:
        parts.append(descs[0][:400])
    return _tokens(" ".join(parts))


def _record_names(rec) -> dict[str, str]:
    """Normalised name → display name for a record's label and aliases."""
    out: dict[str, str] = {}
    for n in [getattr(rec, "label", "") or ""] + list(getattr(rec, "aliases", None) or []):
        k = normalize_name(n)
        if len(k) >= 3 and k not in out:
            out[k] = n
    return out


# ── Candidate generation ─────────────────────────────────────────────


def _chunks(records: list, max_cluster: int) -> list[list]:
    ordered = sorted(records, key=lambda r: normalize_name(getattr(r, "label", "") or ""))
    return [
        ordered[i:i + max_cluster]
        for i in range(0, len(ordered), max_cluster)
        if len(ordered[i:i + max_cluster]) >= 2
    ]


def find_candidate_clusters(
    record_set,
    *,
    min_token_overlap: int = 2,
    max_cluster: int = 12,
    max_org_records: int = 60,
) -> list[CandidateCluster]:
    """Group records that may describe the same entity.

    Two kinds of groups are produced, then de-duplicated:

    - organisation groups: records sharing a normalised organisation,
      sub-clustered by label/alias/description token overlap;
    - shared-name groups: records whose labels or aliases coincide.

    Groups larger than ``max_cluster`` are split into name-sorted chunks.
    Organisations with more than ``max_org_records`` records are treated
    as too generic to group on.
    """
    records = list(getattr(record_set, "records", None) or [])
    if not records:
        return []
    org_attrs = org_attribute_names(record_set)
    clusters: list[CandidateCluster] = []

    by_org: dict[str, list[int]] = defaultdict(list)
    org_display: dict[str, str] = {}
    for idx, rec in enumerate(records):
        for value in _record_org_values(rec, org_attrs):
            for key in normalize_org(value):
                if idx not in by_org[key]:
                    by_org[key].append(idx)
                org_display.setdefault(key, value)

    sigs = [_signature_tokens(r) for r in records]
    for key, idxs in by_org.items():
        if len(idxs) < 2 or len(idxs) > max_org_records:
            continue
        parent = {i: i for i in idxs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a_pos, a in enumerate(idxs):
            for b in idxs[a_pos + 1:]:
                if len(sigs[a] & sigs[b]) >= min_token_overlap:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
        groups: dict[int, list] = defaultdict(list)
        for i in idxs:
            groups[find(i)].append(records[i])
        for group in groups.values():
            for chunk in _chunks(group, max_cluster):
                clusters.append(CandidateCluster(
                    organization=org_display.get(key, key),
                    records=chunk,
                    reason=f"they share the organisation '{org_display.get(key, key)}'",
                ))

    by_name: dict[str, list[int]] = defaultdict(list)
    name_display: dict[str, str] = {}
    for idx, rec in enumerate(records):
        for key, display in _record_names(rec).items():
            by_name[key].append(idx)
            name_display.setdefault(key, display)
    for key, idxs in by_name.items():
        if len(idxs) < 2:
            continue
        group = [records[i] for i in idxs]
        for chunk in _chunks(group, max_cluster):
            clusters.append(CandidateCluster(
                organization="",
                records=chunk,
                reason=f"they share the name '{name_display[key]}'",
            ))

    # Drop exact repeats and groups fully contained in a larger group.
    keyed = [(frozenset(id(r) for r in c.records), c) for c in clusters]
    keyed.sort(key=lambda kc: -len(kc[0]))
    kept: list[tuple[frozenset, CandidateCluster]] = []
    for ids, cluster in keyed:
        if any(ids <= other for other, _ in kept):
            continue
        kept.append((ids, cluster))
    order = {id(c): i for i, c in enumerate(clusters)}
    return sorted((c for _, c in kept), key=lambda c: order[id(c)])


def find_clusters(
    record_set, *, min_token_overlap: int = 2, max_cluster: int = 12
) -> list[tuple[str, list]]:
    """Backward-compatible wrapper returning ``(organization, records)``."""
    return [
        (c.organization or c.reason, c.records)
        for c in find_candidate_clusters(
            record_set, min_token_overlap=min_token_overlap, max_cluster=max_cluster,
        )
    ]


# ── LLM review ───────────────────────────────────────────────────────


def _source_urls(rec, limit: int = 4) -> list[str]:
    urls: list[str] = []
    for av in (getattr(rec, "attributes", None) or {}).values():
        for sv in getattr(av, "values", None) or []:
            for src in getattr(sv, "sources", None) or []:
                url = (getattr(src, "url", "") or "").split("?")[0]
                if url and url not in urls:
                    urls.append(url)
                if len(urls) >= limit:
                    return urls
    return urls


def _format_record(rec, idx: int, max_desc_chars: int, org_attrs: list[str] | None = None) -> str:
    label = getattr(rec, "label", "") or ""
    aliases = list(getattr(rec, "aliases", None) or [])
    descs = _first_values(rec, _DESCRIPTION_ATTRS)[:2]
    desc = " || ".join(d[:max_desc_chars] for d in descs) or "(no description)"
    orgs = _record_org_values(rec, list(org_attrs or _LEGACY_ORG_ATTRS))
    tech = _attr_value(rec, "Technology Type") or "(unknown)"
    func = ", ".join(_first_values(rec, _FUNCTION_ATTRS)) or "(unknown)"
    lines = [
        f"  [{idx}] label: {label}",
        f"      aliases: {', '.join(aliases) if aliases else '(none)'}",
        f"      organisation: {'; '.join(orgs) if orgs else '(unknown)'}",
        f"      tech_type: {tech}",
        f"      function: {func}",
        f"      description: {desc}",
        f"      sources: {', '.join(_source_urls(rec)) or '(none)'}",
    ]
    return "\n".join(lines)


async def _audit_one(
    llm,
    cluster: CandidateCluster,
    sem: asyncio.Semaphore,
    max_desc_chars: int,
    *,
    category: str = "",
    org_attrs: list[str] | None = None,
) -> DedupAuditResult:
    member_labels = [getattr(r, "label", "") or "" for r in cluster.records]
    records_block = "\n".join(
        _format_record(r, i, max_desc_chars, org_attrs) for i, r in enumerate(cluster.records)
    )
    async with sem:
        try:
            result = await llm.structured_completion(
                prompt=_PROMPT,
                response_format=_SCHEMA,
                variables={
                    "category": category or "real-world entities",
                    "reason": cluster.reason,
                    "n_records": len(cluster.records),
                    "records_block": records_block,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return DedupAuditResult(
                organization=cluster.organization,
                member_labels=member_labels,
                candidate_reason=cluster.reason,
                error=str(exc),
            )
    by_upper = {lab.upper(): lab for lab in member_labels}
    merge_sets: list[MergeSet] = []
    for ms in (result.get("merge_sets") or []):
        if not isinstance(ms, dict):
            continue
        canonical = str(ms.get("canonical_label") or "").strip()
        members: list[str] = []
        for m in ms.get("member_labels") or []:
            exact = by_upper.get(str(m).strip().upper())
            if exact and exact not in members:
                members.append(exact)
        if canonical.upper() not in {m.upper() for m in members}:
            # LLM proposed a canonical label that isn't a member;
            # fall back to the first member.
            if not members:
                continue
            canonical = members[0]
        else:
            canonical = by_upper[canonical.upper()]
        if len(members) < 2:
            continue
        merge_sets.append(MergeSet(
            canonical_label=canonical,
            member_labels=members,
            rationale=str(ms.get("rationale") or "").strip(),
        ))
    aliases_by_label = {
        (getattr(r, "label", "") or "").upper(): {a.upper(): a for a in (getattr(r, "aliases", None) or [])}
        for r in cluster.records
    }
    corrections: list[AliasCorrection] = []
    for ac in (result.get("alias_corrections") or []):
        if not isinstance(ac, dict):
            continue
        label = by_upper.get(str(ac.get("record_label") or "").strip().upper())
        alias = aliases_by_label.get((label or "").upper(), {}).get(str(ac.get("alias") or "").strip().upper())
        if not label or not alias:
            continue
        belongs = by_upper.get(str(ac.get("belongs_to_label") or "").strip().upper(), "")
        corrections.append(AliasCorrection(
            record_label=label,
            alias=alias,
            belongs_to_label=belongs if belongs != label else "",
            rationale=str(ac.get("rationale") or "").strip(),
        ))
    return DedupAuditResult(
        organization=cluster.organization,
        member_labels=member_labels,
        candidate_reason=cluster.reason,
        confidence=float(result.get("confidence") or 0.0),
        reason=str(result.get("reason") or ""),
        merge_sets=merge_sets,
        alias_corrections=corrections,
    )


async def audit_clusters(
    record_set,
    llm,
    *,
    concurrency: int = 8,
    max_desc_chars: int = 400,
    min_token_overlap: int = 2,
    max_cluster: int = 12,
    clusters: list[CandidateCluster] | None = None,
    progress_cb=None,
    on_result=None,
) -> list[DedupAuditResult]:
    """Audit every candidate group in ``record_set``.

    Returns one ``DedupAuditResult`` per group. Groups with no proposals
    still appear in the output so callers can see what was considered.
    ``on_result(cluster, result)`` is called as each group completes.
    """
    if llm is None:
        return []
    if clusters is None:
        clusters = find_candidate_clusters(
            record_set,
            min_token_overlap=min_token_overlap,
            max_cluster=max_cluster,
        )
    if not clusters:
        return []
    category = getattr(record_set, "category", "") or ""
    org_attrs = org_attribute_names(record_set)
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def run(cluster: CandidateCluster):
        result = await _audit_one(
            llm, cluster, sem, max_desc_chars, category=category, org_attrs=org_attrs,
        )
        return cluster, result

    tasks = [asyncio.create_task(run(c)) for c in clusters]
    results: list[DedupAuditResult] = []
    for i, t in enumerate(asyncio.as_completed(tasks), start=1):
        cluster, result = await t
        results.append(result)
        if on_result is not None:
            on_result(cluster, result)
        if progress_cb is not None:
            try:
                progress_cb(i, len(tasks))
            except Exception:  # noqa: BLE001
                pass
    return results


def flagged_results(
    results: Iterable[DedupAuditResult], *, confidence_threshold: float = 0.7
) -> list[DedupAuditResult]:
    """Filter results down to those with at least one proposal."""
    return [
        r for r in results
        if r.error is None
        and (r.merge_sets or r.alias_corrections)
        and r.confidence >= confidence_threshold
    ]


# ── Applying reviewed proposals ──────────────────────────────────────


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def find_record(record_set, label: str):
    """Find a record by label, else by alias (case-insensitive)."""
    target = (label or "").strip().casefold()
    if not target:
        return None
    records = getattr(record_set, "records", None) or []
    for r in records:
        if (r.label or "").strip().casefold() == target:
            return r
    for r in records:
        if any((a or "").strip().casefold() == target for a in (r.aliases or [])):
            return r
    return None


def resolve_record(record_set, label: str):
    """Find the record a reviewed proposal refers to.

    Matches labels case-insensitively; a label absorbed by an earlier merge is
    followed through ``merge_duplicates`` history to the record that absorbed
    it. Aliases are never used, because a misattached alias would redirect the
    change to the wrong record.
    """
    target = (label or "").strip().casefold()
    records = getattr(record_set, "records", None) or []
    history = getattr(record_set, "history", None) or []
    seen: set[str] = set()
    while target and target not in seen:
        seen.add(target)
        for r in records:
            if (r.label or "").strip().casefold() == target:
                return r
        target = next(
            (
                (h.get("label") or "").strip().casefold()
                for h in reversed(history)
                if h.get("op") == "merge_duplicates"
                and any((a or "").strip().casefold() == target for a in h.get("absorbed") or [])
            ),
            "",
        )
    return None


def apply_alias_correction(
    record_set,
    record_label: str,
    alias: str,
    *,
    belongs_to_label: str = "",
    rationale: str = "",
    applied_at: str | None = None,
) -> bool:
    """Remove a misattached ``alias`` from a record and log it in history."""
    rec = resolve_record(record_set, record_label)
    target = (alias or "").strip()
    if rec is None or not target or target.upper() == (rec.label or "").upper():
        return False
    from .models import clean_alias

    wanted = {target.casefold(), clean_alias(target).casefold()} - {""}

    def matches(name: str) -> bool:
        return (name or "").casefold() in wanted or clean_alias(name).casefold() in wanted

    keys = [k for k in rec.alias_counts if matches(k) and k.casefold() != (rec.label or "").casefold()]
    before = len(rec.aliases)
    for k in keys:
        rec.alias_counts.pop(k, None)
    rec.aliases = [a for a in rec.aliases if not matches(a)]
    if not keys and len(rec.aliases) == before:
        return False
    rec.updated_at = datetime.now()
    record_set.history.append({
        "op": "remove_alias",
        "applied_at": applied_at or _now(),
        "label": rec.label,
        "alias": target,
        "belongs_to": belongs_to_label,
        "note": rationale,
        "source": "duplicate audit",
    })
    return True


def apply_merge_set(
    record_set,
    canonical_label: str,
    member_labels: list[str],
    *,
    rationale: str = "",
    applied_at: str | None = None,
    primaries: list | None = None,
) -> list[str]:
    """Merge every member into the canonical record and log it in history.

    Members are resolved with :func:`resolve_record` (label, or the record
    that absorbed that label in an earlier merge; never by alias), so a
    member already absorbed is skipped. Pinned labels win: if the
    canonical record is not pinned but exactly one member was pinned (by
    the user or an earlier merge), that member's label is kept. The
    resulting label is pinned so absorbed alias counts cannot rename it.
    Returns the labels of absorbed records; the surviving record is
    appended to ``primaries`` when given.
    """
    canonical = (canonical_label or "").strip()
    primary = resolve_record(record_set, canonical)
    if primary is None:
        return []
    others = []
    for label in member_labels:
        other = resolve_record(record_set, label)
        if other is not None and other is not primary and other not in others:
            others.append(other)
    if not others:
        return []
    pinned = [o for o in others if o.manual_canonical]
    if not primary.manual_canonical and len(pinned) == 1:
        # A member's label was pinned by the user or an earlier merge: keep it.
        others = [primary if o is pinned[0] else o for o in others]
        primary = pinned[0]
    if primary.label.upper() != canonical.upper() and not primary.manual_canonical:
        # The canonical name is currently an alias of the primary: promote it,
        # unless the label was pinned by the user or an earlier merge.
        old = primary.label
        primary.alias_counts.setdefault(old, 1)
        primary.label = canonical
        primary.aliases = [a for a in primary.aliases if a.upper() != canonical.upper()]
        if old and old.upper() not in {a.upper() for a in primary.aliases}:
            primary.aliases.append(old)
    primary.manual_canonical = True
    absorbed: list[str] = []
    for other in others:
        primary.merge_from(other)
        try:
            record_set.records.remove(other)
        except ValueError:
            continue
        absorbed.append(other.label)
    if absorbed:
        if primaries is not None and primary not in primaries:
            primaries.append(primary)
        if getattr(record_set, "do_not_merge", None):
            names = {primary.label, *absorbed}
            record_set.do_not_merge = {p for p in record_set.do_not_merge if not set(p) <= names}
        record_set.history.append({
            "op": "merge_duplicates",
            "applied_at": applied_at or _now(),
            "label": primary.label,
            "absorbed": absorbed,
            "note": rationale,
            "source": "duplicate audit",
        })
    return absorbed


def apply_results(
    record_set,
    results: Iterable[DedupAuditResult],
    *,
    confidence_threshold: float = 0.7,
    applied_at: str | None = None,
) -> dict:
    """Apply every proposal in ``results`` at or above the threshold.

    Alias corrections are applied before merges so misattached names are
    not carried into merged records.
    """
    applied_at = applied_at or _now()
    accepted = flagged_results(results, confidence_threshold=confidence_threshold)
    stats = {"aliases_removed": 0, "merge_sets": 0, "records_absorbed": 0}
    for r in accepted:
        for c in r.alias_corrections:
            if apply_alias_correction(
                record_set, c.record_label, c.alias,
                belongs_to_label=c.belongs_to_label, rationale=c.rationale,
                applied_at=applied_at,
            ):
                stats["aliases_removed"] += 1
    for r in accepted:
        for m in r.merge_sets:
            absorbed = apply_merge_set(
                record_set, m.canonical_label, m.member_labels,
                rationale=m.rationale, applied_at=applied_at,
            )
            if absorbed:
                stats["merge_sets"] += 1
                stats["records_absorbed"] += len(absorbed)
    return stats


# ── Reviewable proposal tables ───────────────────────────────────────

PROPOSAL_FIELDS = [
    "proposal_id", "kind", "approve", "confidence", "group_reason",
    "canonical_label", "member_labels", "record_label", "alias",
    "belongs_to_label", "rationale",
]


def proposal_rows(
    results: Iterable[DedupAuditResult], *, confidence_threshold: float = 0.7
) -> list[dict]:
    """Flatten audit results into one reviewable row per proposal.

    ``approve`` is pre-filled with ``yes`` at or above the threshold and
    left blank otherwise, so reviewers can edit the column before applying.
    """
    rows: list[dict] = []
    for r in results:
        if r.error:
            continue
        approve = "yes" if r.confidence >= confidence_threshold else ""
        for m in r.merge_sets:
            rows.append({
                "kind": "merge", "approve": approve, "confidence": r.confidence,
                "group_reason": r.candidate_reason,
                "canonical_label": m.canonical_label,
                "member_labels": "\n".join(m.member_labels),
                "record_label": "", "alias": "", "belongs_to_label": "",
                "rationale": m.rationale,
            })
        for c in r.alias_corrections:
            rows.append({
                "kind": "remove_alias", "approve": approve, "confidence": r.confidence,
                "group_reason": r.candidate_reason,
                "canonical_label": "", "member_labels": "",
                "record_label": c.record_label, "alias": c.alias,
                "belongs_to_label": c.belongs_to_label, "rationale": c.rationale,
            })
    rows.sort(key=lambda row: (row["kind"] != "remove_alias", row["canonical_label"] or row["record_label"]))
    for i, row in enumerate(rows, start=1):
        row["proposal_id"] = i
    return [{k: row.get(k, "") for k in PROPOSAL_FIELDS} for row in rows]


def apply_proposal_rows(
    record_set,
    rows: Iterable[dict],
    *,
    applied_at: str | None = None,
    primaries: list | None = None,
) -> list[dict]:
    """Apply rows whose ``approve`` column is yes/true/1.

    Alias removals run before merges. Returns one status row per approved
    proposal (``applied``, ``skipped`` with a reason). Records that absorbed
    others are appended to ``primaries`` when given.
    """
    applied_at = applied_at or _now()
    approved = [
        r for r in rows
        if str(r.get("approve", "")).strip().casefold() in {"yes", "y", "true", "1"}
    ]
    approved.sort(key=lambda r: r.get("kind") != "remove_alias")
    out: list[dict] = []
    for row in approved:
        kind = row.get("kind")
        status = {"proposal_id": row.get("proposal_id"), "kind": kind}
        if kind == "remove_alias":
            ok = apply_alias_correction(
                record_set, row.get("record_label", ""), row.get("alias", ""),
                belongs_to_label=row.get("belongs_to_label", ""),
                rationale=row.get("rationale", ""), applied_at=applied_at,
            )
            status.update(
                label=row.get("record_label", ""), detail=row.get("alias", ""),
                status="applied" if ok else "skipped: record or alias not found",
            )
        elif kind == "merge":
            members = [m.strip() for m in str(row.get("member_labels", "")).splitlines() if m.strip()]
            survivors: list = []
            absorbed = apply_merge_set(
                record_set, row.get("canonical_label", ""), members,
                rationale=row.get("rationale", ""), applied_at=applied_at,
                primaries=survivors,
            )
            if primaries is not None:
                primaries.extend(s for s in survivors if s not in primaries)
            status.update(
                label=survivors[0].label if survivors else row.get("canonical_label", ""),
                detail="\n".join(absorbed),
                status="applied" if absorbed else "skipped: nothing left to merge",
            )
        else:
            status.update(label="", detail="", status=f"skipped: unknown kind {kind!r}")
        out.append(status)
    return out
