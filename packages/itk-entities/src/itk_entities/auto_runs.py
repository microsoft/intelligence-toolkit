# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Unattended runs: the auto-mode loop and the rediscovery benchmark."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time

from . import config, inputs, state, storage
from .state import ResearchProgress, Session, UsageStats

_logger = logging.getLogger(__name__)


def start_auto_mode(
    session: Session,
    api_key: str,
    category: str,
    guidance: str = "",
    schema_attributes: list[dict] | None = None,
    *,
    reference_labels: list[str] | None = None,
    search_languages: list[str] | None = None,
    target_language: str = "English",
    max_iterations: int = 5,
    per_iter_query_budget: int = 30,
    normalize_every: int = 3,
    min_iterations: int = 2,
    concurrency: int = 5,
    model: str = config.DEFAULT_MODEL,
    budget: float = 10.0,
) -> None:
    """Run auto mode in a daemon background thread.

    Drives ``run_agentic`` → ``verify_unverified`` → ``normalize``
    in a loop until an LLM judge declares completion or the
    iteration cap is hit. Optional ``reference_labels`` seed the
    record set with entity names (values are NOT trusted — fresh
    verification is required). ``search_languages`` enables
    multilingual query fan-out via :mod:`multilingual`.
    """
    if state.is_running(session):
        return

    session.stop_auto = False
    session.progress = ResearchProgress(
        is_running=True,
        stage="Starting auto mode…",
        max_iterations=int(max_iterations),
    )
    session.df = None
    session.dataset_json = None

    session.runs_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
    ts = time.strftime("%Y%m%d-%H%M%S")
    session.run_dir = session.runs_dir / f"{ts}_auto_{safe}"
    session.run_dir.mkdir(parents=True, exist_ok=True)
    session.usage_baseline = UsageStats()

    def _run() -> None:
        try:
            from itk_schemify import Schemify
            from itk_schemify.models import (
                SchemaAttribute,
                SchemifyConfig,
            )

            from .auto import run_auto_loop
            from .multilingual import make_query_translator

            cfg = SchemifyConfig(
                api_key=api_key,
                search_model=model,
                completion_model=model,
                max_budget=budget,
                cache_enabled=True,
                target_language=(target_language or "English").strip() or "English",
            )
            session.schemify = Schemify(cfg)

            # Wire the multilingual translator after the LLM client
            # exists so it can share the same cache.
            if search_languages:
                cfg.query_translator = make_query_translator(
                    session.schemify.llm,
                    list(search_languages),
                    cache=session.schemify.cache,
                )

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
                    state.bump_record_version(session)
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

                if reference_labels:
                    labels = inputs.dedupe_preserving_order(reference_labels)
                    added = 0
                    if labels:
                        from itk_schemify.models import (
                            Record,
                        )

                        rs = session.schemify.record_set
                        existing = {
                            (r.label or "").strip().casefold() for r in rs.records
                        }
                        for name in labels:
                            if not name or name.casefold() in existing:
                                continue
                            rec = Record(label=name.upper())
                            ok, _ = rs.add_record(rec, use_fuzzy=False)
                            if ok:
                                added += 1
                                existing.add(name.casefold())
                        # Bias prompts away from blindly trusting the
                        # reference values — only the labels are seeds.
                        base = (rs.guidance or "").rstrip()
                        note = (
                            "Auto-mode reference dataset: the entity names "
                            "below were seeded from a prior partial dataset "
                            "as research candidates ONLY. Their attribute "
                            "values are NOT trusted — re-verify everything "
                            "from fresh web sources, and discover NEW "
                            "entities beyond the seed list. Seed entities "
                            f"({len(labels)}): "
                            + ", ".join(labels[:50])
                            + (" …" if len(labels) > 50 else "")
                        )
                        rs.guidance = f"{base}\n\n{note}" if base else note
                        state.bump_record_version(session)
                    session.progress.stage = f"Seeded {added} reference entities…"

                def _auto_progress(phase: str, info: dict) -> None:
                    session.progress.sub_phase = phase
                    iter_n = int(info.get("iteration", 0) or 0)
                    max_n = int(info.get("max_iterations", max_iterations) or 0)
                    if iter_n:
                        session.progress.iteration = iter_n
                    if max_n:
                        session.progress.max_iterations = max_n
                    if "entity_count" in info:
                        session.progress.entity_count = int(
                            info.get("entity_count") or 0
                        )
                    if phase == "Done":
                        session.progress.stage = "Auto mode finishing…"
                        session.progress.stop_reason = str(
                            info.get("stop_reason") or ""
                        )
                    else:
                        label = f"{phase}"
                        if iter_n and max_n:
                            label += f" (iter {iter_n}/{max_n})"
                        session.progress.stage = label

                async def _judge(rs, history):
                    from .auto import judge_completeness

                    verdict = await judge_completeness(
                        session.schemify.llm, rs, history
                    )
                    session.progress.judge_complete = bool(verdict.complete)
                    session.progress.judge_confidence = float(verdict.confidence)
                    session.progress.judge_reason = verdict.reason or ""
                    session.progress.judge_missing_gaps = list(
                        verdict.missing_gaps or []
                    )
                    return verdict

                session.progress.stage = "Auto-mode loop starting…"
                result = loop.run_until_complete(
                    run_auto_loop(
                        session.schemify,
                        max_iterations=int(max_iterations),
                        per_iter_query_budget=int(per_iter_query_budget),
                        concurrency=int(concurrency),
                        output_dir=str(session.run_dir) if session.run_dir else None,
                        progress_callback=_auto_progress,
                        judge_callback=_judge,
                        should_stop=lambda: session.stop_auto,
                        normalize_every=int(normalize_every),
                        min_iterations=int(min_iterations),
                    )
                )

                session.progress.iteration_history = [
                    {
                        "iteration": h.index,
                        "new_entities": h.new_entities,
                        "total_entities": h.total_entities,
                        "queries_run": h.queries_run,
                        "cost_usd": round(h.cost_usd, 4),
                        "phase_split": list(h.phase_split),
                        "judge": h.judge.to_dict() if h.judge else None,
                    }
                    for h in result.per_iteration_history
                ]
                session.progress.stop_reason = result.stop_reason

                session.df = session.schemify.to_dataframe()
                session.dataset_json = storage.build_dataset_json(session)

                stats = session.schemify.get_stats()
                llm_usage = stats.get("llm_usage", {}) or {}
                session.usage = UsageStats(
                    total_tokens=int(llm_usage.get("total_tokens", 0) or 0),
                    total_cost_usd=float(llm_usage.get("total_cost_usd", 0.0) or 0.0),
                    queries_run=session.progress.query_count,
                )

                session.progress.is_running = False
                session.progress.is_complete = True
                session.progress.stage = f"Auto mode complete ({result.stop_reason})"
                if session.schemify.record_set:
                    session.progress.entity_count = len(
                        session.schemify.record_set.records
                    )
                state.bump_record_version(session)

                try:
                    storage.save_run(session, category=category)
                except OSError as e:
                    _logger.warning("_save_run after auto failed: %s", e)
            finally:
                loop.close()

        except Exception as e:
            _logger.exception("auto-mode worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()


def stop_auto_mode(session: Session) -> None:
    """Ask the auto-mode worker to stop at the next iteration boundary."""
    session.stop_auto = True
    session.progress.stage = "Stopping auto mode…"


def start_rediscovery_benchmark(
    session: Session,
    api_key: str,
    category: str,
    guidance: str = "",
    schema_attributes: list[dict] | None = None,
    *,
    reference_labels: list[str],
    search_languages: list[str] | None = None,
    target_language: str = "English",
    max_iterations: int = 4,
    per_iter_query_budget: int = 25,
    gap_fill_query_budget: int = 200,
    gap_fill_cost_reserve_usd: float = 25.0,
    normalize_every: int = 3,
    min_iterations: int = 2,
    concurrency: int = 5,
    model: str = config.DEFAULT_MODEL,
    budget: float = 50.0,
) -> None:
    """Run a blind-rediscovery + gap-fill benchmark in a daemon thread.

    Unlike :meth:`start_auto_mode` (guide mode), the reference labels
    are NOT seeded before research. The auto loop runs blind, then a
    comparison + targeted gap-fill pass attempts to recover any
    reference entities the loop missed, using each missed name as a
    per-entity search anchor. Returns immediately; poll
    ``session.progress`` for status.

    Budget is split: blind discovery is capped at
    ``budget - gap_fill_cost_reserve_usd``; the reserve is held back
    so the gap-fill probe is guaranteed at least that much headroom.
    """
    if state.is_running(session):
        return
    if not reference_labels:
        raise ValueError("start_rediscovery_benchmark requires reference_labels")

    reserve = max(0.0, float(gap_fill_cost_reserve_usd))
    if reserve >= budget:
        raise ValueError(
            f"gap_fill_cost_reserve_usd ({reserve}) must be < budget ({budget})"
        )
    blind_budget = float(budget) - reserve

    session.stop_auto = False
    session.progress = ResearchProgress(
        is_running=True,
        stage="Starting rediscovery benchmark…",
        max_iterations=int(max_iterations),
    )
    session.df = None
    session.dataset_json = None

    session.runs_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
    ts = time.strftime("%Y%m%d-%H%M%S")
    session.run_dir = session.runs_dir / f"{ts}_benchmark_{safe}"
    session.run_dir.mkdir(parents=True, exist_ok=True)
    session.usage_baseline = UsageStats()

    def _run() -> None:
        try:
            from itk_schemify import Schemify
            from itk_schemify.models import (
                SchemaAttribute,
                SchemifyConfig,
            )

            from .auto import (
                gap_fill_against_reference,
                judge_completeness,
                run_auto_loop,
            )
            from .multilingual import make_query_translator

            cfg = SchemifyConfig(
                api_key=api_key,
                search_model=model,
                completion_model=model,
                max_budget=blind_budget,
                cache_enabled=True,
                target_language=(target_language or "English").strip() or "English",
            )
            session.schemify = Schemify(cfg)

            if search_languages:
                cfg.query_translator = make_query_translator(
                    session.schemify.llm,
                    list(search_languages),
                    cache=session.schemify.cache,
                )

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
                    state.bump_record_version(session)
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

                def _auto_progress(phase: str, info: dict) -> None:
                    session.progress.sub_phase = phase
                    iter_n = int(info.get("iteration", 0) or 0)
                    max_n = int(info.get("max_iterations", max_iterations) or 0)
                    if iter_n:
                        session.progress.iteration = iter_n
                    if max_n:
                        session.progress.max_iterations = max_n
                    if "entity_count" in info:
                        session.progress.entity_count = int(
                            info.get("entity_count") or 0
                        )
                    if phase == "Done":
                        session.progress.stage = "Blind discovery finished — comparing…"
                        session.progress.stop_reason = str(
                            info.get("stop_reason") or ""
                        )
                    else:
                        label = f"{phase}"
                        if iter_n and max_n:
                            label += f" (iter {iter_n}/{max_n})"
                        session.progress.stage = label

                async def _judge(rs, history):
                    verdict = await judge_completeness(
                        session.schemify.llm, rs, history
                    )
                    session.progress.judge_complete = bool(verdict.complete)
                    session.progress.judge_confidence = float(verdict.confidence)
                    session.progress.judge_reason = verdict.reason or ""
                    session.progress.judge_missing_gaps = list(
                        verdict.missing_gaps or []
                    )
                    return verdict

                # ── Phase A: blind discovery (NO reference seeds) ──
                session.progress.stage = "Blind discovery loop starting…"
                result = loop.run_until_complete(
                    run_auto_loop(
                        session.schemify,
                        max_iterations=int(max_iterations),
                        per_iter_query_budget=int(per_iter_query_budget),
                        concurrency=int(concurrency),
                        output_dir=str(session.run_dir) if session.run_dir else None,
                        progress_callback=_auto_progress,
                        judge_callback=_judge,
                        should_stop=lambda: session.stop_auto,
                        normalize_every=int(normalize_every),
                        min_iterations=int(min_iterations),
                    )
                )

                session.progress.iteration_history = [
                    {
                        "iteration": h.index,
                        "new_entities": h.new_entities,
                        "total_entities": h.total_entities,
                        "queries_run": h.queries_run,
                        "cost_usd": round(h.cost_usd, 4),
                        "phase_split": list(h.phase_split),
                        "judge": h.judge.to_dict() if h.judge else None,
                    }
                    for h in result.per_iteration_history
                ]
                session.progress.stop_reason = result.stop_reason

                # ── Phase B + C: compare and gap-fill ──
                # Restore full budget so the reserved headroom is
                # available for targeted gap-fill, regardless of how
                # much the blind loop consumed.
                cfg.max_budget = float(budget)

                def _gap_progress(phase: str, info: dict) -> None:
                    session.progress.sub_phase = phase
                    # Iteration counter belongs to the auto loop;
                    # don't pretend gap-fill is part of it.
                    session.progress.iteration = 0
                    session.progress.stage = f"Gap-fill: {phase}"
                    if "reference_count" in info:
                        session.progress.benchmark_reference_count = int(
                            info["reference_count"]
                        )
                    if "rediscovered_before" in info:
                        session.progress.benchmark_rediscovered_before = int(
                            info["rediscovered_before"]
                        )
                    if "fuzzy_matches_before" in info:
                        session.progress.benchmark_fuzzy_matches_before = int(
                            info["fuzzy_matches_before"]
                        )
                    if "missed_before" in info:
                        session.progress.benchmark_missed_before = int(
                            info["missed_before"]
                        )

                gap = loop.run_until_complete(
                    gap_fill_against_reference(
                        session.schemify,
                        reference_labels,
                        gap_fill_query_budget=int(gap_fill_query_budget),
                        concurrency=int(concurrency),
                        output_dir=str(session.run_dir) if session.run_dir else None,
                        progress_callback=_gap_progress,
                        should_stop=lambda: session.stop_auto,
                    )
                )

                session.progress.benchmark_reference_count = gap.reference_count
                session.progress.benchmark_rediscovered_before = gap.rediscovered_before
                session.progress.benchmark_fuzzy_matches_before = (
                    gap.fuzzy_matches_before
                )
                session.progress.benchmark_missed_before = len(gap.missed_before)
                session.progress.benchmark_recovered_via_gap_fill = (
                    gap.recovered_via_gap_fill
                )
                session.progress.benchmark_still_missing = len(gap.still_missing)
                session.progress.benchmark_blind_recall = gap.blind_recall
                session.progress.benchmark_blind_recall_fuzzy = gap.blind_recall_fuzzy
                session.progress.benchmark_total_recall = gap.total_recall
                session.progress.benchmark_still_missing_sample = list(
                    gap.still_missing[:50]
                )

                # Persist the full still-missing list to the run dir.
                if session.run_dir:
                    try:
                        (session.run_dir / "benchmark_report.json").write_text(
                            json.dumps(
                                {
                                    **gap.to_dict(),
                                    "still_missing_full": gap.still_missing,
                                    "stop_reason": result.stop_reason,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except OSError as e:
                        _logger.warning("benchmark_report write failed: %s", e)

                # Final normalize after gap-fill records were added.
                try:
                    loop.run_until_complete(session.schemify.normalize())
                except Exception as e:  # noqa: BLE001
                    _logger.warning("benchmark final normalize failed: %s", e)

                session.df = session.schemify.to_dataframe()
                session.dataset_json = storage.build_dataset_json(session)

                # Pull a fresh live snapshot so query_count reflects
                # ALL phases (blind loop + gap-fill), not just the
                # last on_progress callback fire.
                try:
                    live = session.schemify.get_live_progress() or {}
                    session.progress.query_count = max(
                        int(live.get("query_count", 0) or 0),
                        session.progress.query_count,
                    )
                except Exception as e:  # noqa: BLE001
                    _logger.warning("get_live_progress() at finalize failed: %s", e)

                stats = session.schemify.get_stats()
                llm_usage = stats.get("llm_usage", {}) or {}
                session.usage = UsageStats(
                    total_tokens=int(llm_usage.get("total_tokens", 0) or 0),
                    total_cost_usd=float(llm_usage.get("total_cost_usd", 0.0) or 0.0),
                    queries_run=session.progress.query_count,
                )

                session.progress.is_running = False
                session.progress.is_complete = True
                session.progress.stage = (
                    f"Rediscovery benchmark complete — "
                    f"blind {gap.blind_recall:.0%}, "
                    f"with gap-fill {gap.total_recall:.0%}"
                )
                if session.schemify.record_set:
                    session.progress.entity_count = len(
                        session.schemify.record_set.records
                    )
                state.bump_record_version(session)

                try:
                    storage.save_run(session, category=category)
                except OSError as e:
                    _logger.warning("_save_run after benchmark failed: %s", e)
            finally:
                loop.close()

        except Exception as e:
            _logger.exception("rediscovery-benchmark worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()
