# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Research runs: start, stop and continue discovery, plus guidance and seed entities."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time

from . import config, inputs, state, storage
from .state import ResearchProgress, Session, UsageStats

_logger = logging.getLogger(__name__)


def start_research(
    session: Session,
    api_key: str,
    category: str,
    guidance: str = "",
    schema_attributes: list[dict] | None = None,
    max_queries: int = 30,
    concurrency: int = 5,
    model: str = config.DEFAULT_MODEL,
    budget: float = 10.0,
    verify: bool = True,
    phase_split: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed_labels: list[str] | None = None,
    complete: bool = False,
) -> None:
    """Start research in a daemon background thread.

    ``seed_labels`` are known entity names whose values are researched from scratch;
    ``complete`` fills every record's missing attributes after discovery.
    """
    if state.is_running(session):
        return

    session.progress = ResearchProgress(is_running=True, stage="Starting…")
    session.df = None
    session.dataset_json = None

    # Allocate a run directory up front so per-iteration snapshots and
    # incremental data.json survive crashes.
    session.runs_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
    ts = time.strftime("%Y%m%d-%H%M%S")
    session.run_dir = session.runs_dir / f"{ts}_{safe}"
    session.run_dir.mkdir(parents=True, exist_ok=True)
    session.usage_baseline = UsageStats()

    def _run() -> None:
        try:
            from itk_schemify import Schemify
            from itk_schemify.models import (
                SchemaAttribute,
                SchemifyConfig,
            )

            cfg = SchemifyConfig(
                api_key=api_key,
                search_model=model,
                completion_model=model,
                max_budget=budget,
                cache_enabled=True,
            )
            session.schemify = Schemify(cfg)

            def _on_progress(stage: str, current: int, total: int) -> None:
                session.progress.stage = stage
                session.progress.current = current
                session.progress.total = total
                if session.schemify and session.schemify.record_set:
                    snap = None
                    getter = getattr(session.schemify, "get_live_progress", None)
                    if callable(getter):
                        try:
                            snap = getter()
                        except Exception as e:  # noqa: BLE001
                            _logger.warning("get_live_progress() failed: %s", e)
                    if snap is not None:
                        session.progress.entity_count = int(snap.get("entity_count", 0))
                        session.progress.query_count = max(
                            int(snap.get("query_count", 0)), current
                        )
                    else:
                        session.progress.entity_count = len(
                            session.schemify.record_set.records
                        )
                        history_count = len(
                            getattr(session.schemify, "query_history", []) or []
                        )
                        counter = int(
                            getattr(session.schemify, "_query_counter", 0) or 0
                        )
                        session.progress.query_count = max(
                            history_count, counter, current
                        )
                    state.bump_record_version(session)
                # Throttled disk snapshot so we don't write JSON on every tick.
                try:
                    storage.snapshot_partial(session, category=category, throttle=True)
                except Exception as e:  # noqa: BLE001
                    _logger.warning("snapshot_partial failed: %s", e)

            session.schemify.on_progress(_on_progress)

            sa = None
            if schema_attributes:
                sa = [SchemaAttribute(**a) for a in schema_attributes]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                session.progress.stage = "Initializing schema…"
                loop.run_until_complete(
                    session.schemify.initialize(
                        category=category,
                        guidance=guidance,
                        schema_attributes=sa,
                    )
                )

                session.progress.stage = "Running web search queries…"
                seeds = [
                    {"label": s}
                    for s in inputs.dedupe_preserving_order(seed_labels or [])
                ]
                loop.run_until_complete(
                    session.schemify.run_agentic(
                        max_queries=max_queries,
                        concurrency=concurrency,
                        phase_split=phase_split,
                        output_dir=str(session.run_dir) if session.run_dir else None,
                        seed_records=seeds or None,
                    )
                )

                if complete:
                    session.progress.stage = "Filling missing attributes…"
                    loop.run_until_complete(
                        session.schemify.complete_all(
                            concurrency=concurrency, verbose=False
                        )
                    )

                if verify:
                    # Verification is normally run on demand from the UI
                    # (Review tab). Leaving the flag for callers that want
                    # the legacy behaviour of finishing fully verified.
                    session.progress.stage = "Verifying attribute values…"
                    loop.run_until_complete(
                        session.schemify.verify_unverified(concurrency=concurrency)
                    )

                session.progress.stage = "Finalizing dataset…"
                session.schemify.finalize()

                session.df = session.schemify.to_dataframe()
                # Serialize to dict for JSON export (drops empty schema cols)
                session.dataset_json = storage.build_dataset_json(session)

                # Usage stats
                stats = session.schemify.get_stats()
                llm_usage = stats.get("llm_usage", {})
                session.usage = UsageStats(
                    total_tokens=llm_usage.get("total_tokens", 0),
                    total_cost_usd=llm_usage.get("total_cost_usd", 0.0),
                    queries_run=session.progress.query_count,
                )

                session.progress.is_running = False
                session.progress.is_complete = True
                session.progress.stage = "Complete"
                if session.schemify and session.schemify.record_set:
                    session.progress.entity_count = len(
                        session.schemify.record_set.records
                    )
                state.bump_record_version(session)

                # Persist completed run so the UI can resume it later.
                try:
                    storage.save_run(session, category=category)
                except OSError as e:
                    _logger.warning("_save_run failed: %s", e)
            finally:
                loop.close()

        except Exception as e:
            _logger.exception("research worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()


def stop_research(session: Session) -> None:
    """Snapshot current state and mark as stopped (thread runs to completion)."""
    if session.schemify and session.schemify.record_set:
        try:
            session.df = session.schemify.to_dataframe()
            state.bump_record_version(session)
        except Exception as e:  # noqa: BLE001
            _logger.warning("to_dataframe on stop failed: %s", e)
        # Force-flush any pending throttled snapshot so the user keeps
        # the work-in-progress on disk.
        try:
            storage.snapshot_partial(
                session, category=session.schemify.record_set.category
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("final snapshot on stop failed: %s", e)
    session.progress.is_running = False
    session.progress.stage = "Stopped by user"
    # Auto-mode orchestrator polls this between iterations.
    session.stop_auto = True


def propose_search_languages(
    session: Session,
    api_key: str,
    category: str,
    guidance: str = "",
    model: str = config.DEFAULT_MODEL,
) -> list[dict]:
    """Synchronous wrapper around :func:`multilingual.propose_search_languages`.

    Spins up a one-shot Schemify LLM client to make the call so the
    UI can offer "Suggest source languages" before any research run
    has been started. Returns a list of
    ``{"code", "name", "rationale"}`` dicts.
    """
    from itk_schemify.llm import LLMClient
    from itk_schemify.models import SchemifyConfig

    from .multilingual import propose_search_languages

    cfg = SchemifyConfig(
        api_key=api_key,
        search_model=model,
        completion_model=model,
        cache_enabled=False,
    )
    llm = LLMClient(cfg)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            propose_search_languages(llm, category, guidance)
        )
    finally:
        loop.close()


def continue_research(
    session: Session,
    max_queries: int = 30,
    concurrency: int = 5,
    verify: bool = False,
    phase_split: tuple[float, float, float] = (0.2, 0.2, 0.6),
) -> bool:
    """Run additional research on top of the existing record set.

    Unlike :meth:`start_research` this preserves all current records,
    aliases and exclusions and simply extends the query history. The
    default ``phase_split`` shifts effort toward completion (filling in
    missing attributes for existing entities, including any seeds the
    user added via :meth:`add_candidate_entities`) rather than broad
    discovery.

    Returns False if there's no active Schemify state to continue from.
    """
    if not state.can_continue_research(session):
        return False

    category = (
        session.schemify.record_set.category if session.schemify.record_set else ""
    )
    # Reset completion / error flags but keep entity_count etc.
    session.progress.is_running = True
    session.progress.is_complete = False
    session.progress.error = None
    session.progress.stage = "Continuing research…"
    session.progress.current = 0
    session.progress.total = 0

    def _run() -> None:
        try:

            def _on_progress(stage: str, current: int, total: int) -> None:
                session.progress.stage = stage
                session.progress.current = current
                session.progress.total = total
                if session.schemify and session.schemify.record_set:
                    snap = None
                    getter = getattr(session.schemify, "get_live_progress", None)
                    if callable(getter):
                        try:
                            snap = getter()
                        except Exception as e:  # noqa: BLE001
                            _logger.warning("get_live_progress() failed: %s", e)
                    if snap is not None:
                        session.progress.entity_count = int(snap.get("entity_count", 0))
                        session.progress.query_count = max(
                            int(snap.get("query_count", 0)), current
                        )
                    else:
                        session.progress.entity_count = len(
                            session.schemify.record_set.records
                        )
                        history_count = len(
                            getattr(session.schemify, "query_history", []) or []
                        )
                        counter = int(
                            getattr(session.schemify, "_query_counter", 0) or 0
                        )
                        session.progress.query_count = max(
                            history_count, counter, current
                        )
                    state.bump_record_version(session)
                try:
                    storage.snapshot_partial(session, category=category, throttle=True)
                except Exception as e:  # noqa: BLE001
                    _logger.warning("snapshot_partial failed: %s", e)

            session.schemify.on_progress(_on_progress)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                session.progress.stage = "Running additional web search queries…"
                loop.run_until_complete(
                    session.schemify.run_agentic(
                        max_queries=max_queries,
                        concurrency=concurrency,
                        phase_split=phase_split,
                        output_dir=str(session.run_dir) if session.run_dir else None,
                    )
                )

                if verify:
                    session.progress.stage = "Verifying attribute values…"
                    loop.run_until_complete(
                        session.schemify.verify_unverified(concurrency=concurrency)
                    )

                session.progress.stage = "Finalizing dataset…"
                session.schemify.finalize()

                session.df = session.schemify.to_dataframe()
                session.dataset_json = storage.build_dataset_json(session)

                stats = session.schemify.get_stats()
                llm_usage = stats.get("llm_usage", {})
                session.usage = UsageStats(
                    total_tokens=llm_usage.get("total_tokens", 0),
                    total_cost_usd=llm_usage.get("total_cost_usd", 0.0),
                    queries_run=session.progress.query_count,
                )

                session.progress.is_running = False
                session.progress.is_complete = True
                session.progress.stage = "Complete"
                if session.schemify.record_set:
                    session.progress.entity_count = len(
                        session.schemify.record_set.records
                    )
                state.bump_record_version(session)

                try:
                    storage.save_run(session, category=category)
                except OSError as e:
                    _logger.warning("_save_run after continue failed: %s", e)
            finally:
                loop.close()

        except Exception as e:
            _logger.exception("continue_research worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()
    return True


def append_guidance(session: Session, note: str) -> bool:
    """Append a free-form user note to the record set's research guidance.

    The text is added on its own paragraph prefixed with ``User note:`` so
    it shows up clearly in every downstream prompt that interpolates
    ``{guidance}``. Returns True if anything was appended.
    """
    text = (note or "").strip()
    if not text or not session.schemify or not session.schemify.record_set:
        return False
    rs = session.schemify.record_set
    prefix = (rs.guidance or "").rstrip()
    suffix = f"User note: {text}"
    rs.guidance = f"{prefix}\n\n{suffix}" if prefix else suffix
    session.dataset_json = storage.build_dataset_json(session)
    try:
        storage.snapshot_partial(session, category=rs.category)
    except Exception as e:  # noqa: BLE001
        _logger.warning("snapshot_partial failed: %s", e)
    return True


def add_candidate_entities(session: Session, names: list[str]) -> int:
    """Add user-supplied entity labels as blank seed records.

    Returns the number of records actually added (skips duplicates).
    """
    if not session.schemify or not session.schemify.record_set:
        return 0
    from itk_schemify.models import Record

    added = 0
    existing = {
        (r.label or "").strip().casefold() for r in session.schemify.record_set.records
    }
    for raw in names:
        name = (raw or "").strip()
        if not name or name.casefold() in existing:
            continue
        rec = Record(label=name.upper())
        ok, _ = session.schemify.record_set.add_record(rec, use_fuzzy=False)
        if ok:
            added += 1
            existing.add(name.casefold())
    if added:
        session.df = session.schemify.to_dataframe()
        session.dataset_json = storage.build_dataset_json(session)
        state.bump_record_version(session)
        try:
            storage.snapshot_partial(
                session, category=session.schemify.record_set.category
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot_partial failed: %s", e)
    return added
