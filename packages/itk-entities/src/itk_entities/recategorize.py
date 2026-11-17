# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Recategorization: propose and apply a new taxonomy, fill gaps by search, and the relevance filter."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from . import config, schema, state, storage
from .state import ResearchProgress, Session

_logger = logging.getLogger(__name__)


def propose_recategorization(
    session: Session,
    constraints: str,
    *,
    attributes: list[str] | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Ask an LLM to propose tightened taxonomies for the current dataset.

    No web search is performed here — the model only sees per-attribute
    observed-value frequency tables and the user's ``constraints`` and
    returns a revised schema plus a complete value-translation table.
    Per-record remappings are expanded mechanically from that table.

    Returns the proposal dict (``schema``, ``value_translations``,
    ``record_remappings``, ``out_of_scope_records``, ``unresolved``,
    ``validation_issues``). Nothing is applied — review then call
    :meth:`apply_recategorization`.
    """
    from itk_schemify.propose import run_schema_proposal

    data = storage.build_dataset_json(session)
    if not data:
        raise ValueError("No dataset to re-categorize. Run or load a dataset first.")

    # Prefer the live client's key; fall back to the supplied one.
    key = api_key
    llm = getattr(session.schemify, "llm", None)
    if not key and llm is not None:
        key = getattr(getattr(llm, "config", None), "api_key", None)
    if not key:
        raise ValueError("propose_recategorization requires an OpenAI API key.")

    proposal = run_schema_proposal(
        data,
        constraints=constraints or "",
        target_attrs=attributes or None,
        model=model or config.DEFAULT_MODEL,
        api_key=key,
    )
    return proposal


def apply_recategorization(
    session: Session,
    proposal: dict,
    *,
    remove_out_of_scope: bool = False,
) -> dict:
    """Apply a reviewed proposal to the live dataset in place.

    Remaps existing attribute values to the new canonical set and updates
    each attribute's ``canonical_values`` so subsequent web-search
    re-classification is constrained to the new taxonomy.

    By default records whose target attribute values were *all* dropped
    are kept (with that attribute emptied) so they can be re-classified
    via :meth:`start_recategorization_search`. Set ``remove_out_of_scope``
    to drop them instead.

    Returns a summary dict with counts and, per attribute, the labels of
    records left with no value (``gaps``).
    """
    from itk_schemify.audit import apply_recategorization as _apply
    from itk_schemify.models import RecordSet

    if not session.schemify or not session.schemify.record_set:
        raise ValueError("No live dataset to re-categorize.")

    data = storage.build_dataset_json(session)
    if not data:
        raise ValueError("No dataset to re-categorize.")

    # When gap-filling, keep out-of-scope records so they can be
    # re-searched rather than deleted.
    effective = proposal
    if not remove_out_of_scope and proposal.get("out_of_scope_records"):
        effective = dict(proposal)
        effective["out_of_scope_records"] = []

    new_data = _apply(data, effective)

    rs = RecordSet.from_dict(new_data)
    rs.history.append({
        "op": "recategorize",
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema": proposal.get("schema") or [],
        "value_translations": proposal.get("value_translations") or {},
        "removed_attributes": proposal.get("removed_attributes") or [],
        "remappings_applied": len(proposal.get("record_remappings") or []),
        "removed_records": (
            [r.get("label") for r in (proposal.get("out_of_scope_records") or [])]
            if remove_out_of_scope
            else []
        ),
    })
    session.schemify.record_set = rs
    # Reattach the merge arbiter to the fresh record set (best effort).
    attach = getattr(session.schemify, "_attach_merge_arbiter", None)
    if callable(attach):
        try:
            attach(rs)
        except Exception as e:  # noqa: BLE001
            _logger.warning("re-attach merge arbiter failed: %s", e)

    storage.post_curation_refresh(session)

    target_attrs = [
        s["name"] for s in (proposal.get("schema") or []) if s.get("name")
    ] or [a.name for a in rs.schema_attributes if getattr(a, "is_closed_set", False)]
    gaps = recategorization_gaps(session, target_attrs)
    return {
        "records": len(rs.records),
        "attributes": [a.name for a in rs.schema_attributes],
        "remappings_applied": len(proposal.get("record_remappings") or []),
        "unresolved": len(proposal.get("unresolved") or []),
        "removed_out_of_scope": (
            len(proposal.get("out_of_scope_records") or [])
            if remove_out_of_scope
            else 0
        ),
        "gaps": gaps,
        "gap_total": sum(len(v) for v in gaps.values()),
    }


def recategorization_gaps(
    session: Session, attributes: list[str]
) -> dict[str, list[str]]:
    """Return, per attribute, the labels of records with no value.

    These are the records that pure remapping could not categorize (all
    their old values were dropped or unresolved) — the candidates for
    web-search gap-fill.
    """
    rs = getattr(session.schemify, "record_set", None) if session.schemify else None
    if not rs:
        return {}
    out: dict[str, list[str]] = {}
    for attr in attributes or []:
        missing: list[str] = []
        for r in rs.records:
            av = r.attributes.get(attr)
            if av is None or not getattr(av, "value", ""):
                missing.append(r.label)
        out[attr] = missing
    return out


def _recat_text_blob(session: Session, record) -> str:
    """Lowercased label + aliases + open-set attribute values for keyword
    candidate matching."""
    rs = session.schemify.record_set
    open_attrs = [a.name for a in rs.schema_attributes if not a.is_closed_set]
    parts = [getattr(record, "label", "")] + list(getattr(record, "aliases", []) or [])
    for a in open_attrs:
        av = record.attributes.get(a)
        if av:
            parts += [v.value for v in av.values if v.value]
    return " \n ".join(p for p in parts if p).lower()


def _resolve_recat_candidates(session: Session, attr, scope, keywords, labels=None):
    """Records to re-search for ``attr`` given scope + optional keywords/labels."""
    rs = session.schemify.record_set
    kws = [k.lower() for k in (keywords or []) if k]
    wanted = {l.strip().upper() for l in (labels or []) if l} or None
    out = []
    for r in rs.records:
        if wanted is not None and (r.label or "").upper() not in wanted:
            continue
        av = r.attributes.get(attr)
        has_value = av is not None and getattr(av, "value", "")
        if scope == "gaps" and has_value:
            continue
        if kws:
            blob = _recat_text_blob(session, r)
            if not any(k in blob for k in kws):
                continue
        out.append(r)
    return out


def recategorization_candidates(
    session: Session,
    attributes: list[str],
    *,
    scope: str = "all",
    candidate_keywords: list[str] | None = None,
    labels: list[str] | None = None,
) -> dict[str, list[str]]:
    """Preview which records a web-search pass would touch (per attribute).

    Lets the UI estimate cost before spending. Same selection logic as
    :meth:`start_recategorization_search`.
    """
    rs = getattr(session.schemify, "record_set", None) if session.schemify else None
    if not rs:
        return {}
    scope = (scope or "all").strip().lower()
    return {
        attr: [
            r.label
            for r in _resolve_recat_candidates(
                session, attr, scope, candidate_keywords, labels
            )
        ]
        for attr in (attributes or [])
        if attr
    }


def start_recategorization_search(
    session: Session,
    attributes: list[str],
    *,
    mode: str = "augment",
    only_values: list[str] | None = None,
    candidate_keywords: list[str] | None = None,
    labels: list[str] | None = None,
    scope: str = "all",
    concurrency: int = 4,
) -> None:
    """Re-classify records against the new taxonomy using web search.

    Runs in a background thread. Each targeted record is re-extracted via
    one grounded web search; extraction is constrained to the attribute's
    current ``canonical_values`` (set by :meth:`apply_recategorization`).

    ``mode``:
      - ``"augment"`` (default, zero-regression): the record's existing
        value(s) are preserved; the search result is only *added* when it
        introduces a value that isn't already present. With ``only_values``
        set, only those values are added — ideal for populating genuinely
        new categories (e.g. "Compliance & Inspections") without disturbing
        correct classifications.
      - ``"reassign"``: the existing value is cleared and fully replaced by
        the fresh search result (may churn good values — use deliberately).

    ``candidate_keywords``: restrict the pass to records whose label /
    aliases / description / org fields contain any of these substrings
    (case-insensitive). Cheapest way to target a specific new category.

    ``scope`` (applied on top of keywords): ``"all"`` = every matching
    record; ``"gaps"`` = only records currently empty for the attribute.

    Progress is reported on ``session.progress``; the list of additions is
    available afterwards on ``session.recat_added``.
    """
    if state.is_running(session):
        return
    if not session.schemify or not session.schemify.record_set:
        raise ValueError("No live dataset — load a run with an API key first.")
    if getattr(session.schemify, "llm", None) is None:
        raise ValueError(
            "Web-search re-classification requires a live research session "
            "with an API key."
        )
    attrs = [a for a in (attributes or []) if a]
    if not attrs:
        return

    rs = session.schemify.record_set
    mode = (mode or "augment").strip().lower()
    scope = (scope or "all").strip().lower()
    accept = set(only_values or [])

    work: list[tuple[object, str]] = []
    for attr in attrs:
        for r in _resolve_recat_candidates(
            session, attr, scope, candidate_keywords, labels
        ):
            work.append((r, attr))

    total = len(work)
    session.recat_added = []  # [{"label","attribute","value"}]
    session.progress = ResearchProgress(
        is_running=True,
        stage=f"Re-classifying {total} values via web search…",
        total=total,
        current=0,
        entity_count=len(rs.records),
    )

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _reclassify(
                    session, work, mode=mode, accept=accept, concurrency=concurrency
                )
            )
            storage.post_curation_refresh(session)
            session.progress.is_running = False
            session.progress.is_complete = True
            session.progress.stage = (
                f"Re-classification complete — {len(session.recat_added)} values added"
            )
        except Exception as e:
            _logger.exception("recategorization search worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"
        finally:
            loop.close()

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()


async def _reclassify(
    session: Session,
    work,
    *,
    mode: str,
    accept: set,
    concurrency: int,
    search_focus: str | None = None,
) -> None:
    """Re-extract ``(record, attribute)`` pairs via grounded search; see start_recategorization_search."""
    import copy as _copy

    from itk_schemify.models import AttributeValue

    rs = session.schemify.record_set
    total = len(work)
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    done = [0]

    async def _one(record, attr):
        async with semaphore:
            original = _copy.deepcopy(record.attributes.get(attr))
            record.attributes[attr] = AttributeValue(values=[])
            try:
                await session.schemify.extraction.expand_record(
                    record,
                    rs,
                    target_attributes=[attr],
                    search_focus=search_focus,
                )
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "re-classify %s/%s failed: %s",
                    getattr(record, "label", "?"),
                    attr,
                    e,
                )
            new_av = record.attributes.get(attr)
            new_sourced = {v.value: v.sources for v in new_av.values} if new_av else {}
            if mode == "reassign":
                for val in new_sourced:
                    session.recat_added.append({
                        "label": record.label,
                        "attribute": attr,
                        "value": val,
                    })
            else:
                if original is not None:
                    record.attributes[attr] = original
                else:
                    record.attributes.pop(attr, None)
                for val, srcs in new_sourced.items():
                    if accept and val not in accept:
                        continue
                    if record.attributes.get(attr) is None:
                        record.attributes[attr] = AttributeValue(values=[])
                    record.attributes[attr].add_value_with_sources(val, srcs)
                    session.recat_added.append({
                        "label": record.label,
                        "attribute": attr,
                        "value": val,
                    })
            done[0] += 1
            session.progress.current = done[0]
            session.progress.stage = f"Researching via web search ({done[0]}/{total})"

    await asyncio.gather(*[_one(r, a) for r, a in work])


def start_relevance_filter(
    session: Session,
    *,
    criterion: str,
    evidence_attribute: str,
    connections: list[str],
    evidence_description: str = "",
    research_evidence: bool = True,
    search_focus: str | None = None,
    labels: list[str] | None = None,
    context_attributes: list[str] | None = None,
    extra_evidence_attributes: list[str] | None = None,
    confidence_threshold: float = 0.7,
    concurrency: int = 8,
) -> None:
    """Background: optionally research a sourced evidence attribute for every record,
    then judge each record against ``criterion``. Results land on
    ``session.relevance_results``; nothing is removed (use :meth:`apply_scope_exclusions`).

    ``search_focus`` runs a targeted search for the evidence (not the entity's cached
    general page); ``labels`` restricts research and judging to those records.
    """
    from itk_schemify import scope_audit as _sa

    if (
        state.is_running(session)
        or not session.schemify
        or not session.schemify.record_set
    ):
        return
    if getattr(session.schemify, "llm", None) is None:
        raise ValueError(
            "The relevance filter requires a live session with an API key."
        )
    schema.add_schema_attribute(
        session, evidence_attribute, evidence_description or criterion
    )
    rs = session.schemify.record_set
    session.relevance_results = {}
    session.recat_added = []
    session.progress = ResearchProgress(
        is_running=True,
        stage="Researching evidence…",
        total=len(rs.records),
        entity_count=len(rs.records),
    )

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if research_evidence:
                wanted = {l.casefold() for l in labels} if labels else None
                work = [
                    (r, evidence_attribute)
                    for r in rs.records
                    if wanted is None or (r.label or "").casefold() in wanted
                ]
                loop.run_until_complete(
                    _reclassify(
                        session,
                        work,
                        mode="reassign",
                        accept=set(),
                        concurrency=concurrency,
                        search_focus=search_focus,
                    )
                )
            session.progress.stage = "Judging explicit connections…"

            def _p(done: int, total: int) -> None:
                session.progress.current, session.progress.total = done, total

            results = loop.run_until_complete(
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
                    progress_cb=_p,
                )
            )
            session.relevance_results = {
                "total_records": len(rs.records),
                "results": results,
                "not_explicit": _sa.not_explicit_results(results, confidence_threshold),
                "uncertain": [
                    r
                    for r in results
                    if r["explicit"] is None
                    or (
                        r["explicit"] is False
                        and r["confidence"] < confidence_threshold
                    )
                ],
            }
            storage.post_curation_refresh(session)
            session.progress.is_running = False
            session.progress.is_complete = True
            session.progress.stage = f"Relevance check complete — {len(session.relevance_results['not_explicit'])} without an explicit connection"
        except Exception as e:
            _logger.exception("relevance filter worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"
        finally:
            loop.close()

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()
