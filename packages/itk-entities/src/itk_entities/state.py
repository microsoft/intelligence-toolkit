# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The state of one Build Entity Dataset session, and read-only views of it.

Every stage module is a set of plain functions that take a :class:`Session`.
The session holds the live Schemify instance, the last materialised dataset,
progress and usage counters, and the background worker thread. A re-entrant
lock coordinates the worker thread and the caller polling for progress.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from itk_core.paths import CACHE_PATH

_logger = logging.getLogger(__name__)

#: Default folder for saved runs; override per session with ``Session(runs_dir=...)``.
RUNS_DIR = Path(CACHE_PATH) / "build_entity_dataset" / "runs"


@dataclass
class ResearchProgress:
    """Live progress of a research run."""

    stage: str = "Not started"
    current: int = 0
    total: int = 0
    entity_count: int = 0
    query_count: int = 0
    is_running: bool = False
    is_complete: bool = False
    error: str = ""

    # Auto-mode fields. Left at defaults outside of auto-mode runs so
    # existing UI code that reads ``ResearchProgress`` is unaffected.
    iteration: int = 0
    max_iterations: int = 0
    sub_phase: str = ""
    judge_complete: bool | None = None
    judge_confidence: float = 0.0
    judge_reason: str = ""
    judge_missing_gaps: list[str] = field(default_factory=list)
    iteration_history: list[dict] = field(default_factory=list)
    stop_reason: str = ""

    # Rediscovery-benchmark fields. Populated by
    # :func:`itk_entities.auto_runs.start_rediscovery_benchmark`.
    benchmark_reference_count: int = 0
    benchmark_rediscovered_before: int = 0
    benchmark_fuzzy_matches_before: int = 0
    benchmark_missed_before: int = 0
    benchmark_recovered_via_gap_fill: int = 0
    benchmark_still_missing: int = 0
    benchmark_blind_recall: float = 0.0
    benchmark_blind_recall_fuzzy: float = 0.0
    benchmark_total_recall: float = 0.0
    benchmark_still_missing_sample: list[str] = field(default_factory=list)


@dataclass
class UsageStats:
    """Token and cost summary."""

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    queries_run: int = 0


@dataclass
class Session:
    """Everything one Build Entity Dataset session keeps in memory."""

    #: Folder holding one sub-folder per saved run.
    runs_dir: Path = RUNS_DIR
    #: Live ``itk_schemify.Schemify`` instance, or None before a run / when read-only.
    schemify: Any = None
    progress: ResearchProgress = field(default_factory=ResearchProgress)
    usage: UsageStats = field(default_factory=UsageStats)
    thread: threading.Thread | None = None
    dataset_json: dict | None = None
    df: pd.DataFrame | None = None
    run_dir: Path | None = None
    # Coordinates writes/reads between the background worker and the
    # caller's thread. Re-entrant so worker callbacks that call
    # already-locked helpers don't deadlock.
    state_lock: Any = field(default_factory=threading.RLock, repr=False)
    # Cache key for the materialized DataFrame. The worker bumps
    # ``record_version`` on every record-set mutation; readers rebuild
    # the DataFrame only when ``df_cache_version`` lags behind.
    record_version: int = 0
    df_cache_version: int = -1
    cached_df: pd.DataFrame | None = None
    # Snapshot throttling.
    last_snapshot_time: float = 0.0
    snapshot_pending: bool = False
    # Distinguishes runs that were rehydrated read-only from disk
    # (no API key) so the UI can explain why "Continue research" is
    # unavailable.
    read_only_reason: str | None = None
    # Cooperative cancellation flag for the auto-mode worker. The
    # orchestrator checks it between iterations; setting it does not
    # interrupt an in-flight Schemify call.
    stop_auto: bool = False
    # Values added by the most recent recategorization web-search pass.
    recat_added: list[dict] = field(default_factory=list)
    relevance_results: dict = field(default_factory=dict)
    review_verification_report: dict = field(default_factory=dict)
    review_verification_dir: Path | None = None
    review_verification_active: bool = False
    review_verification_stop: bool = False
    review_verification_progress: ResearchProgress = field(
        default_factory=ResearchProgress
    )
    # Usage already recorded in a reloaded run's meta.json; session usage adds to it.
    usage_baseline: UsageStats = field(default_factory=UsageStats)


def is_running(session: Session) -> bool:
    return session.thread is not None and session.thread.is_alive()


def is_verifying_review(session: Session) -> bool:
    return session.review_verification_active


def current_dataframe(session: Session) -> pd.DataFrame | None:
    """Return a DataFrame snapshot of the live record set.

    Rebuilds only when the underlying record set has changed since the
    last call — the UI polls this on every Streamlit rerun, so a naive
    rebuild was O(records) per poll for no new data.
    """
    with session.state_lock:
        if session.df is not None and not session.df.empty:
            return session.df
        if not session.schemify or not session.schemify.record_set:
            return None
        if (
            session.cached_df is not None
            and session.df_cache_version == session.record_version
        ):
            return session.cached_df
    try:
        df = session.schemify.to_dataframe()
    except Exception as e:  # noqa: BLE001
        _logger.warning("to_dataframe() failed: %s", e)
        return None
    with session.state_lock:
        session.cached_df = df
        session.df_cache_version = session.record_version
    return df


def bump_record_version(session: Session) -> None:
    with session.state_lock:
        session.record_version += 1


def schema_attributes(session: Session) -> list[dict]:
    if session.schemify and session.schemify.record_set:
        return [
            {
                "name": a.name,
                "description": getattr(a, "description", ""),
                "is_closed_set": getattr(a, "is_closed_set", False),
                "is_multi_valued": getattr(a, "is_multi_valued", False),
                "locked": getattr(a, "locked", False),
                "canonical_values": list(getattr(a, "canonical_values", []) or []),
                "canonical_value_descriptions": dict(
                    getattr(a, "canonical_value_descriptions", {}) or {}
                ),
            }
            for a in session.schemify.record_set.schema_attributes
        ]
    if session.dataset_json:
        return [
            {
                "name": a.get("name", ""),
                "description": a.get("description", ""),
                "is_closed_set": a.get("is_closed_set", False),
                "is_multi_valued": a.get("is_multi_valued", False),
                "locked": a.get("locked", False),
                "canonical_values": list(a.get("canonical_values") or []),
                "canonical_value_descriptions": dict(
                    a.get("canonical_value_descriptions") or {}
                ),
            }
            for a in session.dataset_json.get("schema_attributes", [])
        ]
    return []


def refresh_progress(session: Session) -> None:
    """Pull live counters from the running Schemify instance.

    Prefers ``Schemify.get_live_progress()`` (a public O(1) snapshot
    introduced for the ITK wrapper). Falls back to reading the
    legacy private counters when running against an older Schemify.
    """
    if not session.schemify:
        return
    snap = None
    getter = getattr(session.schemify, "get_live_progress", None)
    if callable(getter):
        try:
            snap = getter()
        except Exception as e:  # noqa: BLE001
            _logger.warning("get_live_progress() failed: %s", e)
            snap = None
    if snap is None:
        # Legacy fallback.
        try:
            history_count = len(getattr(session.schemify, "query_history", []) or [])
            counter = int(getattr(session.schemify, "_query_counter", 0) or 0)
            rs = getattr(session.schemify, "record_set", None)
            entity_count = len(rs.records) if rs is not None else 0
            snap = {
                "query_count": max(history_count, counter),
                "entity_count": entity_count,
            }
        except Exception as e:  # noqa: BLE001
            _logger.warning("legacy progress read failed: %s", e)
            return
    session.progress.query_count = max(
        int(snap.get("query_count", 0) or 0), session.progress.query_count
    )
    session.progress.entity_count = int(snap.get("entity_count", 0) or 0)
    # Live cost from the LLM client, if available.
    llm = getattr(session.schemify, "llm", None)
    if llm is not None:
        try:
            session.usage.total_cost_usd = float(getattr(llm, "total_cost", 0.0) or 0.0)
            session.usage.total_tokens = int(getattr(llm, "total_tokens", 0) or 0)
            session.usage.queries_run = session.progress.query_count
        except Exception as e:  # noqa: BLE001
            _logger.warning("llm usage read failed: %s", e)


def reset(session: Session) -> None:
    """Drop all in-memory state.

    Refuses while a background worker is still alive — wiping the
    Schemify reference while the worker holds it would just cause
    confusing callback errors. Stop research first, then reset.
    """
    with session.state_lock:
        if is_running(session):
            _logger.warning("reset() called while research is running; ignoring")
            return
        session.schemify = None
        session.progress = ResearchProgress()
        session.usage = UsageStats()
        session.usage_baseline = UsageStats()
        session.thread = None
        session.dataset_json = None
        session.df = None
        session.run_dir = None
        session.record_version += 1
        session.df_cache_version = -1
        session.cached_df = None
        session.last_snapshot_time = 0.0
        session.read_only_reason = None


def can_continue_research(session: Session) -> bool:
    """True iff the in-memory Schemify state is alive enough to keep researching."""
    return bool(
        session.schemify and session.schemify.record_set and not is_running(session)
    )


def is_read_only(session: Session) -> bool:
    """True when the loaded dataset has no live Schemify instance
    (e.g. a saved run loaded without an API key). Curation and
    continue-research are unavailable in this state."""
    return session.read_only_reason is not None and session.schemify is None


def read_only_reason(session: Session) -> str | None:
    """Human-readable explanation of why the dataset is read-only, or None."""
    return session.read_only_reason if is_read_only(session) else None


def count_unverified(session: Session) -> tuple[int, int]:
    """Return ``(entities_with_unverified, total_unverified_values)``.

    An attribute value is "unverified" when it has zero distinct
    citations (``av.evidence.source_count == 0``).
    """
    if not session.schemify or not session.schemify.record_set:
        return (0, 0)
    entities = 0
    values = 0
    for record in session.schemify.record_set.records:
        had = False
        for bucket in (record.attributes, record.additional_attributes):
            for av in bucket.values():
                if av.evidence.source_count == 0:
                    values += 1
                    had = True
        if had:
            entities += 1
    return (entities, values)


def find_record(session: Session, label: str):
    """Find a record by exact label, else by alias (case-insensitive).

    Labels win over aliases so a record whose label equals another
    record's alias is still addressable."""
    if not session.schemify or not session.schemify.record_set:
        return None
    target = (label or "").strip().casefold()
    if not target:
        return None
    records = session.schemify.record_set.records
    for r in records:
        if (r.label or "").strip().casefold() == target:
            return r
    for r in records:
        for a in r.aliases or []:
            if (a or "").strip().casefold() == target:
                return r
    return None


def count_incomplete(session: Session) -> tuple[int, int]:
    """(records with any empty schema attribute, empty cells) across the dataset."""
    rs = getattr(session.schemify, "record_set", None) if session.schemify else None
    if not rs:
        return (0, 0)
    names = [a.name for a in rs.schema_attributes]
    per = [
        sum(1 for n in names if not (r.attributes.get(n) and r.attributes[n].value))
        for r in rs.records
    ]
    return (sum(1 for p in per if p), sum(per))
