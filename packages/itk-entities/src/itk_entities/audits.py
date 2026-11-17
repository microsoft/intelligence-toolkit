# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Model-assisted audits (harmful content, scope, relevance, merge quality, duplicates) and applying their results."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from . import exclusions, storage
from .state import Session

_logger = logging.getLogger(__name__)


# Max records per LLM call when scanning for harmful content. Each
# batch is one LLM round-trip instead of one per record.
SAFETY_SCAN_BATCH_SIZE = 8
SAFETY_SCAN_CONCURRENCY = 6


DEFAULT_SAFETY_PROMPT = (
    "You are a content-safety classifier. Given the following "
    "structured entity record, identify any concerns across these "
    "categories: hate, harassment, violence, sexual, self-harm, "
    "illegal-activity, sensitive-PII, dangerous-instructions.\n\n"
    "If the record is benign, reply exactly: SAFE.\n"
    "Otherwise reply with one line of comma-separated category labels, "
    "then a newline, then a one-sentence reason.\n\n"
    "RECORD:\n{record}\n"
)


def scan_harmful_content(
    session: Session,
    prompt_template: str | None = None,
    *,
    concurrency: int = SAFETY_SCAN_CONCURRENCY,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Use an LLM to flag potentially harmful values across records.

    Each record's prompt is dispatched in parallel via a thread pool
    (the underlying OpenAI client is synchronous) so a 1k-entity
    dataset finishes in ~``ceil(N/concurrency)`` round-trips instead
    of N serial calls.

    Args:
        prompt_template: Optional override for the classifier prompt.
            Must include a ``{record}`` placeholder where the entity
            fields will be rendered. Falls back to
            :attr:`DEFAULT_SAFETY_PROMPT`.
        concurrency: Max parallel LLM calls.
        progress_callback: Optional ``(done, total)`` reporter.

    Returns a list of ``{"label", "record_index", "categories", "reason",
    "fields"}`` dicts. Records with no concerns are omitted. Findings
    are returned in record-index order regardless of completion order.
    """
    if not session.schemify or not session.schemify.record_set:
        return []
    try:
        from itk_core.client import OpenAIClient
        from itk_core.defaults import DEFAULT_TEMPERATURE
    except ImportError as e:
        _logger.warning("safety-scan: AI client import failed: %s", e)
        return []

    template = (prompt_template or DEFAULT_SAFETY_PROMPT).strip()
    if "{record}" not in template:
        template = template + "\n\nRECORD:\n{record}\n"

    client = OpenAIClient()
    records = list(session.schemify.record_set.records)

    # Pre-render the field dicts so the worker pool only does I/O.
    targets: list[tuple[int, str, dict[str, str]]] = []
    for idx, record in enumerate(records):
        fields: dict[str, str] = {}
        if record.label:
            fields["label"] = record.label
        if record.aliases:
            fields["aliases"] = "; ".join(record.aliases)
        for bucket in (record.attributes, record.additional_attributes):
            for k, av in bucket.items():
                val = getattr(av, "value", None)
                if val:
                    fields[k] = str(val)
        if fields:
            targets.append((idx, record.label or "(unlabeled)", fields))

    if not targets:
        return []

    def _scan_one(item: tuple[int, str, dict[str, str]]) -> dict | None:
        idx, label, fields = item
        content = "\n".join(f"{k}: {v}" for k, v in fields.items())
        prompt = template.format(record=content)
        try:
            resp = client.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=DEFAULT_TEMPERATURE,
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("safety-scan: record %s failed: %s", idx, e)
            return {
                "label": label,
                "record_index": idx,
                "categories": ["error"],
                "reason": f"Scan failed: {e}",
                "fields": fields,
            }
        text = (resp or "").strip()
        if not text or text.upper().startswith("SAFE"):
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cats_line = lines[0] if lines else ""
        reason = lines[1] if len(lines) > 1 else ""
        cats = [c.strip() for c in cats_line.split(",") if c.strip()]
        return {
            "label": label,
            "record_index": idx,
            "categories": cats,
            "reason": reason,
            "fields": fields,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    findings: list[dict] = []
    total = len(targets)
    done = 0
    max_workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, t): t for t in targets}
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if progress_callback:
                try:
                    progress_callback(done, total)
                except Exception as e:  # noqa: BLE001
                    _logger.warning("safety-scan progress callback raised: %s", e)
            if result is not None:
                findings.append(result)
    findings.sort(key=lambda f: f.get("record_index", 0))
    return findings


def audit_scope(
    session: Session,
    *,
    api_key: str | None = None,
    model: str | None = None,
    budget: float = 4.0,
    confidence_threshold: float = 0.7,
    concurrency: int = 6,
    progress_cb=None,
) -> dict:
    """Flag records that aren't instances of the dataset's category
    (reports, policies, campaigns, data sources about the topic, …).

    Returns ``{"results", "flagged", "total_records"}``; nothing is
    removed — apply flagged entries with :meth:`apply_scope_exclusions`.
    """
    from itk_schemify import scope_audit as _sa
    from itk_schemify.llm import LLMClient
    from itk_schemify.models import SchemifyConfig

    if not session.schemify or not session.schemify.record_set:
        return {"results": [], "flagged": [], "total_records": 0}
    rs = session.schemify.record_set
    llm = getattr(session.schemify, "llm", None)
    if llm is None:
        if not api_key:
            raise ValueError("audit_scope requires api_key when no llm is attached")
        cfg_kwargs = {"api_key": api_key, "max_budget": float(budget)}
        if model:
            cfg_kwargs["completion_model"] = model
        llm = LLMClient(SchemifyConfig(**cfg_kwargs))

    results = asyncio.run(
        _sa.audit_scope(rs, llm, concurrency=concurrency, progress_cb=progress_cb)
    )
    return {
        "total_records": len(rs.records),
        "results": results,
        "flagged": _sa.flagged_results(results, confidence_threshold),
        "weak_relevance": _sa.weak_relevance_results(results),
    }


def audit_relevance(
    session: Session,
    *,
    criterion: str,
    evidence_attribute: str,
    connections: list[str],
    context_attributes: list[str] | None = None,
    extra_evidence_attributes: list[str] | None = None,
    labels: list[str] | None = None,
    confidence_threshold: float = 0.7,
    concurrency: int = 6,
    progress_cb=None,
) -> dict:
    """Judge each record's cited evidence against an explicit-connection ``criterion``.

    Returns ``{"results", "not_explicit", "total_records"}``. Nothing is removed;
    pass ``not_explicit`` to :meth:`apply_scope_exclusions` to filter.
    """
    from itk_schemify import scope_audit as _sa

    if not session.schemify or not session.schemify.record_set:
        return {"results": [], "not_explicit": [], "total_records": 0}
    if getattr(session.schemify, "llm", None) is None:
        raise ValueError("audit_relevance requires a live session with an API key")
    rs = session.schemify.record_set
    results = asyncio.run(
        _sa.audit_relevance(
            rs,
            session.schemify.llm,
            criterion=criterion,
            evidence_attribute=evidence_attribute,
            connections=connections,
            context_attributes=context_attributes,
            extra_evidence_attributes=extra_evidence_attributes,
            labels=labels,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )
    )
    return {
        "total_records": len(rs.records),
        "results": results,
        "not_explicit": _sa.not_explicit_results(results, confidence_threshold),
    }


def apply_scope_exclusions(session: Session, entries: list[dict]) -> int:
    """Exclude flagged records (also blocks their re-discovery). Returns count removed.

    Removes only the record whose label matches exactly; other records that merely
    carry the label as an alias are left alone.
    """
    if not session.schemify or not session.schemify.record_set:
        return 0
    rs = session.schemify.record_set
    targets: set[str] = set()
    for e in entries or []:
        label = (e.get("label") or "").strip()
        if not label:
            continue
        reason = f"Out of scope ({e.get('entity_kind', 'Other')}): {e.get('reason', '')}".strip()
        exclusions.add_label_exclusion(
            session, label, reason=reason, remove_existing=False
        )
        targets.add(label.casefold())
    before = len(rs.records)
    rs.records = [
        r for r in rs.records if (r.label or "").strip().casefold() not in targets
    ]
    removed = before - len(rs.records)
    if removed:
        storage.post_curation_refresh(session)
    return removed


def audit_merge_quality(
    session: Session,
    *,
    api_key: str | None = None,
    budget: float = 4.0,
    model: str | None = None,
    concurrency: int = 8,
    confidence_threshold: float = 0.7,
    progress_cb=None,
) -> dict:
    """Run the LLM-backed merge-quality audit.

    Returns ``{"results": [...], "flagged": [...], "candidates": int,
    "total_records": int}``. The ``flagged`` list is filtered to
    results the auditor recommends splitting at the given confidence.
    Each entry is the dict form of :class:`AuditResult` so the UI
    can render it directly.
    """
    from itk_schemify import merge_audit as _ma
    from itk_schemify.llm import LLMClient
    from itk_schemify.models import SchemifyConfig

    if not session.schemify or not session.schemify.record_set:
        return {
            "results": [],
            "flagged": [],
            "candidates": 0,
            "total_records": 0,
        }
    rs = session.schemify.record_set

    # Reuse the active llm if we have one; otherwise build a fresh
    # client from the supplied key. We need an api_key one way or
    # the other — the audit makes one LLM call per candidate.
    llm = getattr(session.schemify, "llm", None)
    if llm is None:
        if not api_key:
            raise ValueError(
                "audit_merge_quality requires api_key when no llm is attached"
            )
        cfg_kwargs = {"api_key": api_key, "max_budget": float(budget)}
        if model:
            cfg_kwargs["model"] = model
        llm = LLMClient(SchemifyConfig(**cfg_kwargs))

    async def _run() -> list:
        return await _ma.audit_records(
            rs,
            llm,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )

    results = asyncio.run(_run())
    flagged = _ma.flagged_results(results, confidence_threshold=confidence_threshold)
    return {
        "total_records": len(rs.records),
        "candidates": len(results),
        "flagged": [r.to_dict() for r in flagged],
        "results": [r.to_dict() for r in results],
    }


def apply_audit_split(session: Session, label: str, audit_entry: dict) -> int:
    """Apply a single split proposal (an entry from the audit's
    ``flagged`` list) to the record carrying ``label``. Returns the
    number of new records created (0 if no change).
    """
    from itk_schemify import merge_audit as _ma

    if not session.schemify or not session.schemify.record_set:
        return 0
    rs = session.schemify.record_set
    target = None
    target_idx = -1
    for i, r in enumerate(rs.records):
        if (getattr(r, "label", "") or "").strip().upper() == (
            label or ""
        ).strip().upper():
            target = r
            target_idx = i
            break
    if target is None:
        return 0

    proposals = [
        _ma.SplitProposal(
            label=str(s.get("label") or "").strip(),
            description_indices=[int(i) for i in (s.get("description_indices") or [])],
            rationale=str(s.get("rationale") or ""),
        )
        for s in (audit_entry.get("split_proposal") or [])
        if isinstance(s, dict)
    ]
    if len(proposals) < 2:
        return 0
    result = _ma.AuditResult(
        label=audit_entry.get("label") or label,
        descriptions=list(audit_entry.get("descriptions") or []),
        split_proposal=proposals,
    )
    new_records, dnm_pairs = _ma.apply_split(target, result)
    if len(new_records) <= 1:
        return 0
    # Replace original with the splits in-place to preserve order.
    rs.records[target_idx : target_idx + 1] = new_records
    # Lock the splits as do_not_merge so a future pass won't reunite.
    if not hasattr(rs, "do_not_merge") or rs.do_not_merge is None:
        rs.do_not_merge = set()
    for a, b in dnm_pairs:
        if a and b and a != b:
            rs.do_not_merge.add(frozenset({a, b}))
    storage.post_curation_refresh(session)
    return len(new_records)


def audit_duplicate_clusters(
    session: Session,
    *,
    api_key: str | None = None,
    budget: float = 4.0,
    model: str | None = None,
    concurrency: int = 8,
    confidence_threshold: float = 0.7,
    min_token_overlap: int = 2,
    progress_cb=None,
) -> dict:
    """Run the LLM-backed dedup audit (find under-merged clusters).

    Returns ``{"results": [...], "flagged": [...], "clusters": int,
    "total_records": int}``. Each entry is the dict form of
    :class:`DedupAuditResult`.
    """
    from itk_schemify import dedup_audit as _da
    from itk_schemify.llm import LLMClient
    from itk_schemify.models import SchemifyConfig

    if not session.schemify or not session.schemify.record_set:
        return {
            "results": [],
            "flagged": [],
            "clusters": 0,
            "total_records": 0,
        }
    rs = session.schemify.record_set

    llm = getattr(session.schemify, "llm", None)
    if llm is None:
        if not api_key:
            raise ValueError(
                "audit_duplicate_clusters requires api_key when no llm is attached"
            )
        cfg_kwargs = {"api_key": api_key, "max_budget": float(budget)}
        if model:
            cfg_kwargs["model"] = model
        llm = LLMClient(SchemifyConfig(**cfg_kwargs))

    async def _run() -> list:
        return await _da.audit_clusters(
            rs,
            llm,
            concurrency=concurrency,
            min_token_overlap=min_token_overlap,
            progress_cb=progress_cb,
        )

    results = asyncio.run(_run())
    flagged = _da.flagged_results(results, confidence_threshold=confidence_threshold)
    return {
        "total_records": len(rs.records),
        "clusters": len(results),
        "flagged": [r.to_dict() for r in flagged],
        "results": [r.to_dict() for r in results],
    }


def apply_dedup_merge(
    session: Session,
    canonical_label: str,
    member_labels: list[str],
    rationale: str = "",
) -> int:
    """Collapse every label in ``member_labels`` (other than
    ``canonical_label``) into the canonical record and log the merge
    in the dataset history. Returns the number of records absorbed.
    """
    from itk_schemify import dedup_audit as _da

    if not canonical_label or not member_labels:
        return 0
    if not session.schemify or not session.schemify.record_set:
        return 0
    absorbed = _da.apply_merge_set(
        session.schemify.record_set,
        canonical_label,
        member_labels,
        rationale=rationale,
    )
    if absorbed:
        storage.post_curation_refresh(session)
    return len(absorbed)


def apply_alias_correction(
    session: Session,
    label: str,
    alias: str,
    *,
    belongs_to_label: str = "",
    rationale: str = "",
) -> bool:
    """Remove a misattached alias flagged by the duplicate audit (logged in history)."""
    from itk_schemify import dedup_audit as _da

    if not session.schemify or not session.schemify.record_set:
        return False
    ok = _da.apply_alias_correction(
        session.schemify.record_set,
        label,
        alias,
        belongs_to_label=belongs_to_label,
        rationale=rationale,
    )
    if ok:
        storage.post_curation_refresh(session)
    return ok
