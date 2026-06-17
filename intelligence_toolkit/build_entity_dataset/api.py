# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build Entity Dataset API - wraps Schemify for ITK integration."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from intelligence_toolkit.helpers.constants import CACHE_PATH

from . import config

_logger = logging.getLogger(__name__)

_RUNS_DIR = Path(CACHE_PATH) / "build_entity_dataset" / "runs"

# Minimum seconds between disk snapshots during a running extraction.
# Progress callbacks fire on every query/extraction tick; without
# throttling that's a blocking JSON write per tick.
_SNAPSHOT_MIN_INTERVAL_SECONDS = 10.0

# Max labels per LLM call when asking for alias-merge suggestions.
# Larger datasets are chunked across multiple calls instead of being
# silently truncated.
_ALIAS_SUGGEST_CHUNK_SIZE = 200

# Max records per LLM call when scanning for harmful content. Each
# batch is one LLM round-trip instead of one per record.
_SAFETY_SCAN_BATCH_SIZE = 8
_SAFETY_SCAN_CONCURRENCY = 6


def _parse_alias_suggestions(raw: str) -> list[dict]:
    """Best-effort JSON extraction from an LLM reply.

    The model is asked to return a JSON array of merge suggestions but
    occasionally wraps it in prose or fenced code blocks; tolerate that.
    Returns a list of plain dicts (no validation against the dataset —
    callers do that).
    """
    text = (raw or "").strip()
    if not text:
        return []
    # Strip leading/trailing markdown code fences if present.
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag) and the
        # trailing fence.
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        # Fall back to locating the first '[' ... ']' span.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


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
    judge_complete: Optional[bool] = None
    judge_confidence: float = 0.0
    judge_reason: str = ""
    judge_missing_gaps: list[str] = field(default_factory=list)
    iteration_history: list[dict] = field(default_factory=list)
    stop_reason: str = ""

    # Rediscovery-benchmark fields. Populated by
    # :meth:`BuildEntityDataset.start_rediscovery_benchmark`.
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


class BuildEntityDataset:
    """
    ITK wrapper around Schemify.

    Runs entity extraction in a background thread so Streamlit can poll
    for progress on each rerun without blocking the UI.
    """

    def __init__(self) -> None:
        self._schemify = None
        self.progress = ResearchProgress()
        self.usage: UsageStats = UsageStats()
        self._thread: Optional[threading.Thread] = None
        self._dataset_json: Optional[dict] = None
        self._df: Optional[pd.DataFrame] = None
        self._run_dir: Optional[Path] = None
        # Coordinates writes/reads between the background worker and the
        # Streamlit UI thread. Re-entrant so worker callbacks that call
        # already-locked helpers don't deadlock.
        self._state_lock = threading.RLock()
        # Cache key for the materialized DataFrame. Worker bumps
        # ``_record_version`` on every record-set mutation; readers rebuild
        # the DF only when ``_df_cache_version`` lags behind.
        self._record_version: int = 0
        self._df_cache_version: int = -1
        self._cached_df: Optional[pd.DataFrame] = None
        # Snapshot throttling.
        self._last_snapshot_time: float = 0.0
        self._snapshot_pending: bool = False
        # Distinguishes runs that were rehydrated read-only from disk
        # (no API key) so the UI can explain why "Continue research" is
        # unavailable.
        self._read_only_reason: Optional[str] = None
        # Cooperative cancellation flag for the auto-mode worker. The
        # orchestrator checks it between iterations; setting it does not
        # interrupt an in-flight Schemify call.
        self._stop_auto: bool = False

    # ── Public properties ──────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def dataset_json(self) -> Optional[dict]:
        return self._dataset_json

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        return self._df

    @property
    def run_dir(self) -> Optional[Path]:
        return self._run_dir

    def current_dataframe(self) -> Optional[pd.DataFrame]:
        """Return a DataFrame snapshot of the live record set.

        Rebuilds only when the underlying record set has changed since the
        last call — the UI polls this on every Streamlit rerun, so a naive
        rebuild was O(records) per poll for no new data.
        """
        with self._state_lock:
            if self._df is not None and not self._df.empty:
                return self._df
            if not self._schemify or not self._schemify.record_set:
                return None
            if (
                self._cached_df is not None
                and self._df_cache_version == self._record_version
            ):
                return self._cached_df
        try:
            df = self._schemify.to_dataframe()
        except Exception as e:  # noqa: BLE001
            _logger.warning("to_dataframe() failed: %s", e)
            return None
        with self._state_lock:
            self._cached_df = df
            self._df_cache_version = self._record_version
        return df

    def _bump_record_version(self) -> None:
        with self._state_lock:
            self._record_version += 1

    @property
    def schema_attributes(self) -> list[dict]:
        if self._schemify and self._schemify.record_set:
            return [
                {
                    "name": a.name,
                    "description": getattr(a, "description", ""),
                    "is_closed_set": getattr(a, "is_closed_set", False),
                }
                for a in self._schemify.record_set.schema_attributes
            ]
        if self._dataset_json:
            return [
                {"name": a.get("name", ""), "description": a.get("description", "")}
                for a in self._dataset_json.get("schema_attributes", [])
            ]
        return []

    def refresh_progress(self) -> None:
        """Pull live counters from the running Schemify instance.

        Prefers ``Schemify.get_live_progress()`` (a public O(1) snapshot
        introduced for the ITK wrapper). Falls back to reading the
        legacy private counters when running against an older Schemify.
        """
        if not self._schemify:
            return
        snap = None
        getter = getattr(self._schemify, "get_live_progress", None)
        if callable(getter):
            try:
                snap = getter()
            except Exception as e:  # noqa: BLE001
                _logger.warning("get_live_progress() failed: %s", e)
                snap = None
        if snap is None:
            # Legacy fallback.
            try:
                history_count = len(getattr(self._schemify, "query_history", []) or [])
                counter = int(getattr(self._schemify, "_query_counter", 0) or 0)
                rs = getattr(self._schemify, "record_set", None)
                entity_count = len(rs.records) if rs is not None else 0
                snap = {
                    "query_count": max(history_count, counter),
                    "entity_count": entity_count,
                }
            except Exception as e:  # noqa: BLE001
                _logger.warning("legacy progress read failed: %s", e)
                return
        self.progress.query_count = max(
            int(snap.get("query_count", 0) or 0), self.progress.query_count
        )
        self.progress.entity_count = int(snap.get("entity_count", 0) or 0)
        # Live cost from the LLM client, if available.
        llm = getattr(self._schemify, "llm", None)
        if llm is not None:
            try:
                self.usage.total_cost_usd = float(getattr(llm, "total_cost", 0.0) or 0.0)
                self.usage.total_tokens = int(getattr(llm, "total_tokens", 0) or 0)
                self.usage.queries_run = self.progress.query_count
            except Exception as e:  # noqa: BLE001
                _logger.warning("llm usage read failed: %s", e)

    # ── Research lifecycle ─────────────────────────────────────

    def reset(self) -> None:
        """Drop all in-memory state.

        Refuses while a background worker is still alive — wiping the
        Schemify reference while the worker holds it would just cause
        confusing callback errors. Stop research first, then reset.
        """
        with self._state_lock:
            if self.is_running:
                _logger.warning("reset() called while research is running; ignoring")
                return
            self._schemify = None
            self.progress = ResearchProgress()
            self.usage = UsageStats()
            self._thread = None
            self._dataset_json = None
            self._df = None
            self._run_dir = None
            self._record_version += 1
            self._df_cache_version = -1
            self._cached_df = None
            self._last_snapshot_time = 0.0
            self._read_only_reason = None

    def start_research(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: Optional[list[dict]] = None,
        max_queries: int = 30,
        concurrency: int = 5,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
        verify: bool = True,
        phase_split: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> None:
        """Start research in a daemon background thread."""
        if self.is_running:
            return

        self.progress = ResearchProgress(is_running=True, stage="Starting…")
        self._df = None
        self._dataset_json = None

        # Allocate a run directory up front so per-iteration snapshots and
        # incremental data.json survive crashes.
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
        ts = time.strftime("%Y%m%d-%H%M%S")
        self._run_dir = _RUNS_DIR / f"{ts}_{safe}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        def _run() -> None:
            try:
                from intelligence_toolkit.schemify import Schemify  # noqa: PLC0415
                from intelligence_toolkit.schemify.models import SchemifyConfig, SchemaAttribute  # noqa: PLC0415

                cfg = SchemifyConfig(
                    api_key=api_key,
                    search_model=model,
                    completion_model=model,
                    max_budget=budget,
                    cache_enabled=True,
                )
                self._schemify = Schemify(cfg)

                def _on_progress(stage: str, current: int, total: int) -> None:
                    self.progress.stage = stage
                    self.progress.current = current
                    self.progress.total = total
                    if self._schemify and self._schemify.record_set:
                        snap = None
                        getter = getattr(self._schemify, "get_live_progress", None)
                        if callable(getter):
                            try:
                                snap = getter()
                            except Exception as e:  # noqa: BLE001
                                _logger.warning("get_live_progress() failed: %s", e)
                        if snap is not None:
                            self.progress.entity_count = int(snap.get("entity_count", 0))
                            self.progress.query_count = max(
                                int(snap.get("query_count", 0)), current
                            )
                        else:
                            self.progress.entity_count = len(self._schemify.record_set.records)
                            history_count = len(
                                getattr(self._schemify, "query_history", []) or []
                            )
                            counter = int(
                                getattr(self._schemify, "_query_counter", 0) or 0
                            )
                            self.progress.query_count = max(history_count, counter, current)
                        self._bump_record_version()
                    # Throttled disk snapshot so we don't write JSON on every tick.
                    try:
                        self._snapshot_partial(category=category, throttle=True)
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("snapshot_partial failed: %s", e)

                self._schemify.on_progress(_on_progress)

                sa = None
                if schema_attributes:
                    sa = [SchemaAttribute(**a) for a in schema_attributes]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    self.progress.stage = "Initializing schema…"
                    loop.run_until_complete(
                        self._schemify.initialize(
                            category=category,
                            guidance=guidance,
                            schema_attributes=sa,
                        )
                    )

                    self.progress.stage = "Running web search queries…"
                    loop.run_until_complete(
                        self._schemify.run_agentic(
                            max_queries=max_queries,
                            concurrency=concurrency,
                            phase_split=phase_split,
                            output_dir=str(self._run_dir) if self._run_dir else None,
                        )
                    )

                    if verify:
                        # Verification is normally run on demand from the UI
                        # (Review tab). Leaving the flag for callers that want
                        # the legacy behaviour of finishing fully verified.
                        self.progress.stage = "Verifying attribute values…"
                        loop.run_until_complete(
                            self._schemify.verify_unverified(concurrency=concurrency)
                        )

                    self.progress.stage = "Finalizing dataset…"
                    self._schemify.finalize()

                    self._df = self._schemify.to_dataframe()
                    # Serialize to dict for JSON export (drops empty schema cols)
                    self._dataset_json = self._build_dataset_json()

                    # Usage stats
                    stats = self._schemify.get_stats()
                    llm_usage = stats.get("llm_usage", {})
                    self.usage = UsageStats(
                        total_tokens=llm_usage.get("total_tokens", 0),
                        total_cost_usd=llm_usage.get("total_cost_usd", 0.0),
                        queries_run=self.progress.query_count,
                    )

                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = "Complete"
                    if self._schemify and self._schemify.record_set:
                        self.progress.entity_count = len(
                            self._schemify.record_set.records
                        )
                    self._bump_record_version()

                    # Persist completed run so the UI can resume it later.
                    try:
                        self._save_run(category=category)
                    except OSError as e:
                        _logger.warning("_save_run failed: %s", e)
                finally:
                    loop.close()

            except Exception as e:  # noqa: BLE001
                _logger.exception("research worker failed")
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop_research(self) -> None:
        """Snapshot current state and mark as stopped (thread runs to completion)."""
        if self._schemify and self._schemify.record_set:
            try:
                self._df = self._schemify.to_dataframe()
                self._bump_record_version()
            except Exception as e:  # noqa: BLE001
                _logger.warning("to_dataframe on stop failed: %s", e)
            # Force-flush any pending throttled snapshot so the user keeps
            # the work-in-progress on disk.
            try:
                self._snapshot_partial(
                    category=self._schemify.record_set.category
                )
            except Exception as e:  # noqa: BLE001
                _logger.warning("final snapshot on stop failed: %s", e)
        self.progress.is_running = False
        self.progress.stage = "Stopped by user"
        # Auto-mode orchestrator polls this between iterations.
        self._stop_auto = True

    # ── Auto mode ───────────────────────────────────────────────

    @staticmethod
    def reference_labels_from_file(filename: str, raw: bytes) -> list[str]:
        """Parse an uploaded reference dataset and return the entity labels.

        Accepts the same formats as :meth:`parse_candidate_file` and
        additionally recognises the full dataset JSON shape written by
        :meth:`_save_run` (``{"records": [{"label": ...}, ...]}``).
        Duplicates are deduplicated case-insensitively while preserving
        input order.
        """
        if not raw:
            return []
        name = (filename or "").lower()
        if name.endswith(".json"):
            try:
                text = raw.decode("utf-8-sig", errors="replace")
                data = json.loads(text)
            except Exception:  # noqa: BLE001
                data = None
            if isinstance(data, dict) and isinstance(data.get("records"), list):
                names = []
                for r in data["records"]:
                    if isinstance(r, dict):
                        v = r.get("label") or r.get("name")
                        if isinstance(v, str) and v.strip():
                            names.append(v.strip())
                return BuildEntityDataset._dedupe_preserving_order(names)

        names = BuildEntityDataset.parse_candidate_file(filename, raw)
        return BuildEntityDataset._dedupe_preserving_order(names)

    @staticmethod
    def _dedupe_preserving_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in items:
            v = (raw or "").strip()
            if not v:
                continue
            k = v.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
        return out

    def propose_search_languages(
        self,
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
        from intelligence_toolkit.schemify.llm import LLMClient  # noqa: PLC0415
        from intelligence_toolkit.schemify.models import SchemifyConfig  # noqa: PLC0415
        from .multilingual import propose_search_languages  # noqa: PLC0415

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

    def start_auto_mode(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: Optional[list[dict]] = None,
        *,
        reference_labels: Optional[list[str]] = None,
        search_languages: Optional[list[str]] = None,
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
        if self.is_running:
            return

        self._stop_auto = False
        self.progress = ResearchProgress(
            is_running=True,
            stage="Starting auto mode…",
            max_iterations=int(max_iterations),
        )
        self._df = None
        self._dataset_json = None

        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
        ts = time.strftime("%Y%m%d-%H%M%S")
        self._run_dir = _RUNS_DIR / f"{ts}_auto_{safe}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        def _run() -> None:
            try:
                from intelligence_toolkit.schemify import Schemify  # noqa: PLC0415
                from intelligence_toolkit.schemify.models import (  # noqa: PLC0415
                    SchemifyConfig,
                    SchemaAttribute,
                )
                from .auto import run_auto_loop  # noqa: PLC0415
                from .multilingual import make_query_translator  # noqa: PLC0415

                cfg = SchemifyConfig(
                    api_key=api_key,
                    search_model=model,
                    completion_model=model,
                    max_budget=budget,
                    cache_enabled=True,
                    target_language=(target_language or "English").strip() or "English",
                )
                self._schemify = Schemify(cfg)

                # Wire the multilingual translator after the LLM client
                # exists so it can share the same cache.
                if search_languages:
                    cfg.query_translator = make_query_translator(
                        self._schemify.llm,
                        list(search_languages),
                        cache=self._schemify.cache,
                    )

                def _on_progress(stage: str, current: int, total: int) -> None:
                    self.progress.stage = stage
                    self.progress.current = current
                    self.progress.total = total
                    if self._schemify and self._schemify.record_set:
                        snap = None
                        getter = getattr(self._schemify, "get_live_progress", None)
                        if callable(getter):
                            try:
                                snap = getter()
                            except Exception as e:  # noqa: BLE001
                                _logger.warning("get_live_progress() failed: %s", e)
                        if snap is not None:
                            self.progress.entity_count = int(
                                snap.get("entity_count", 0)
                            )
                            self.progress.query_count = max(
                                int(snap.get("query_count", 0)), current
                            )
                        else:
                            self.progress.entity_count = len(
                                self._schemify.record_set.records
                            )
                        self._bump_record_version()
                    try:
                        self._snapshot_partial(category=category, throttle=True)
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("snapshot_partial failed: %s", e)

                self._schemify.on_progress(_on_progress)

                sa = None
                if schema_attributes:
                    sa = [SchemaAttribute(**a) for a in schema_attributes]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    self.progress.stage = "Initializing schema…"
                    loop.run_until_complete(
                        self._schemify.initialize(
                            category=category,
                            guidance=guidance,
                            schema_attributes=sa,
                        )
                    )

                    if reference_labels:
                        labels = self._dedupe_preserving_order(reference_labels)
                        added = 0
                        if labels:
                            from intelligence_toolkit.schemify.models import (  # noqa: PLC0415
                                Record,
                            )
                            rs = self._schemify.record_set
                            existing = {
                                (r.label or "").strip().casefold()
                                for r in rs.records
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
                            self._bump_record_version()
                        self.progress.stage = (
                            f"Seeded {added} reference entities…"
                        )

                    def _auto_progress(phase: str, info: dict) -> None:
                        self.progress.sub_phase = phase
                        iter_n = int(info.get("iteration", 0) or 0)
                        max_n = int(info.get("max_iterations", max_iterations) or 0)
                        if iter_n:
                            self.progress.iteration = iter_n
                        if max_n:
                            self.progress.max_iterations = max_n
                        if "entity_count" in info:
                            self.progress.entity_count = int(
                                info.get("entity_count") or 0
                            )
                        if phase == "Done":
                            self.progress.stage = "Auto mode finishing…"
                            self.progress.stop_reason = str(
                                info.get("stop_reason") or ""
                            )
                        else:
                            label = f"{phase}"
                            if iter_n and max_n:
                                label += f" (iter {iter_n}/{max_n})"
                            self.progress.stage = label

                    async def _judge(rs, history):
                        from .auto import judge_completeness  # noqa: PLC0415
                        verdict = await judge_completeness(
                            self._schemify.llm, rs, history
                        )
                        self.progress.judge_complete = bool(verdict.complete)
                        self.progress.judge_confidence = float(verdict.confidence)
                        self.progress.judge_reason = verdict.reason or ""
                        self.progress.judge_missing_gaps = list(
                            verdict.missing_gaps or []
                        )
                        return verdict

                    self.progress.stage = "Auto-mode loop starting…"
                    result = loop.run_until_complete(
                        run_auto_loop(
                            self._schemify,
                            max_iterations=int(max_iterations),
                            per_iter_query_budget=int(per_iter_query_budget),
                            concurrency=int(concurrency),
                            output_dir=str(self._run_dir) if self._run_dir else None,
                            progress_callback=_auto_progress,
                            judge_callback=_judge,
                            should_stop=lambda: self._stop_auto,
                            normalize_every=int(normalize_every),
                            min_iterations=int(min_iterations),
                        )
                    )

                    self.progress.iteration_history = [
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
                    self.progress.stop_reason = result.stop_reason

                    self._df = self._schemify.to_dataframe()
                    self._dataset_json = self._build_dataset_json()

                    stats = self._schemify.get_stats()
                    llm_usage = stats.get("llm_usage", {}) or {}
                    self.usage = UsageStats(
                        total_tokens=int(llm_usage.get("total_tokens", 0) or 0),
                        total_cost_usd=float(
                            llm_usage.get("total_cost_usd", 0.0) or 0.0
                        ),
                        queries_run=self.progress.query_count,
                    )

                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = (
                        f"Auto mode complete ({result.stop_reason})"
                    )
                    if self._schemify.record_set:
                        self.progress.entity_count = len(
                            self._schemify.record_set.records
                        )
                    self._bump_record_version()

                    try:
                        self._save_run(category=category)
                    except OSError as e:
                        _logger.warning("_save_run after auto failed: %s", e)
                finally:
                    loop.close()

            except Exception as e:  # noqa: BLE001
                _logger.exception("auto-mode worker failed")
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop_auto_mode(self) -> None:
        """Ask the auto-mode worker to stop at the next iteration boundary."""
        self._stop_auto = True
        self.progress.stage = "Stopping auto mode…"

    def start_rediscovery_benchmark(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: Optional[list[dict]] = None,
        *,
        reference_labels: list[str],
        search_languages: Optional[list[str]] = None,
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
        ``self.progress`` for status.

        Budget is split: blind discovery is capped at
        ``budget - gap_fill_cost_reserve_usd``; the reserve is held back
        so the gap-fill probe is guaranteed at least that much headroom.
        """
        if self.is_running:
            return
        if not reference_labels:
            raise ValueError("start_rediscovery_benchmark requires reference_labels")

        reserve = max(0.0, float(gap_fill_cost_reserve_usd))
        if reserve >= budget:
            raise ValueError(
                f"gap_fill_cost_reserve_usd ({reserve}) must be < budget ({budget})"
            )
        blind_budget = float(budget) - reserve

        self._stop_auto = False
        self.progress = ResearchProgress(
            is_running=True,
            stage="Starting rediscovery benchmark…",
            max_iterations=int(max_iterations),
        )
        self._df = None
        self._dataset_json = None

        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
        ts = time.strftime("%Y%m%d-%H%M%S")
        self._run_dir = _RUNS_DIR / f"{ts}_benchmark_{safe}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        def _run() -> None:
            try:
                from intelligence_toolkit.schemify import Schemify  # noqa: PLC0415
                from intelligence_toolkit.schemify.models import (  # noqa: PLC0415
                    SchemifyConfig,
                    SchemaAttribute,
                )
                from .auto import (  # noqa: PLC0415
                    run_auto_loop,
                    gap_fill_against_reference,
                    judge_completeness,
                )
                from .multilingual import make_query_translator  # noqa: PLC0415

                cfg = SchemifyConfig(
                    api_key=api_key,
                    search_model=model,
                    completion_model=model,
                    max_budget=blind_budget,
                    cache_enabled=True,
                    target_language=(target_language or "English").strip() or "English",
                )
                self._schemify = Schemify(cfg)

                if search_languages:
                    cfg.query_translator = make_query_translator(
                        self._schemify.llm,
                        list(search_languages),
                        cache=self._schemify.cache,
                    )

                def _on_progress(stage: str, current: int, total: int) -> None:
                    self.progress.stage = stage
                    self.progress.current = current
                    self.progress.total = total
                    if self._schemify and self._schemify.record_set:
                        snap = None
                        getter = getattr(self._schemify, "get_live_progress", None)
                        if callable(getter):
                            try:
                                snap = getter()
                            except Exception as e:  # noqa: BLE001
                                _logger.warning("get_live_progress() failed: %s", e)
                        if snap is not None:
                            self.progress.entity_count = int(
                                snap.get("entity_count", 0)
                            )
                            self.progress.query_count = max(
                                int(snap.get("query_count", 0)), current
                            )
                        else:
                            self.progress.entity_count = len(
                                self._schemify.record_set.records
                            )
                        self._bump_record_version()
                    try:
                        self._snapshot_partial(category=category, throttle=True)
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("snapshot_partial failed: %s", e)

                self._schemify.on_progress(_on_progress)

                sa = None
                if schema_attributes:
                    sa = [SchemaAttribute(**a) for a in schema_attributes]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    self.progress.stage = "Initializing schema…"
                    loop.run_until_complete(
                        self._schemify.initialize(
                            category=category,
                            guidance=guidance,
                            schema_attributes=sa,
                        )
                    )

                    def _auto_progress(phase: str, info: dict) -> None:
                        self.progress.sub_phase = phase
                        iter_n = int(info.get("iteration", 0) or 0)
                        max_n = int(info.get("max_iterations", max_iterations) or 0)
                        if iter_n:
                            self.progress.iteration = iter_n
                        if max_n:
                            self.progress.max_iterations = max_n
                        if "entity_count" in info:
                            self.progress.entity_count = int(
                                info.get("entity_count") or 0
                            )
                        if phase == "Done":
                            self.progress.stage = "Blind discovery finished — comparing…"
                            self.progress.stop_reason = str(
                                info.get("stop_reason") or ""
                            )
                        else:
                            label = f"{phase}"
                            if iter_n and max_n:
                                label += f" (iter {iter_n}/{max_n})"
                            self.progress.stage = label

                    async def _judge(rs, history):
                        verdict = await judge_completeness(
                            self._schemify.llm, rs, history
                        )
                        self.progress.judge_complete = bool(verdict.complete)
                        self.progress.judge_confidence = float(verdict.confidence)
                        self.progress.judge_reason = verdict.reason or ""
                        self.progress.judge_missing_gaps = list(
                            verdict.missing_gaps or []
                        )
                        return verdict

                    # ── Phase A: blind discovery (NO reference seeds) ──
                    self.progress.stage = "Blind discovery loop starting…"
                    result = loop.run_until_complete(
                        run_auto_loop(
                            self._schemify,
                            max_iterations=int(max_iterations),
                            per_iter_query_budget=int(per_iter_query_budget),
                            concurrency=int(concurrency),
                            output_dir=str(self._run_dir) if self._run_dir else None,
                            progress_callback=_auto_progress,
                            judge_callback=_judge,
                            should_stop=lambda: self._stop_auto,
                            normalize_every=int(normalize_every),
                            min_iterations=int(min_iterations),
                        )
                    )

                    self.progress.iteration_history = [
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
                    self.progress.stop_reason = result.stop_reason

                    # ── Phase B + C: compare and gap-fill ──
                    # Restore full budget so the reserved headroom is
                    # available for targeted gap-fill, regardless of how
                    # much the blind loop consumed.
                    cfg.max_budget = float(budget)

                    def _gap_progress(phase: str, info: dict) -> None:
                        self.progress.sub_phase = phase
                        # Iteration counter belongs to the auto loop;
                        # don't pretend gap-fill is part of it.
                        self.progress.iteration = 0
                        self.progress.stage = f"Gap-fill: {phase}"
                        if "reference_count" in info:
                            self.progress.benchmark_reference_count = int(
                                info["reference_count"]
                            )
                        if "rediscovered_before" in info:
                            self.progress.benchmark_rediscovered_before = int(
                                info["rediscovered_before"]
                            )
                        if "fuzzy_matches_before" in info:
                            self.progress.benchmark_fuzzy_matches_before = int(
                                info["fuzzy_matches_before"]
                            )
                        if "missed_before" in info:
                            self.progress.benchmark_missed_before = int(
                                info["missed_before"]
                            )

                    gap = loop.run_until_complete(
                        gap_fill_against_reference(
                            self._schemify,
                            reference_labels,
                            gap_fill_query_budget=int(gap_fill_query_budget),
                            concurrency=int(concurrency),
                            output_dir=str(self._run_dir) if self._run_dir else None,
                            progress_callback=_gap_progress,
                            should_stop=lambda: self._stop_auto,
                        )
                    )

                    self.progress.benchmark_reference_count = gap.reference_count
                    self.progress.benchmark_rediscovered_before = gap.rediscovered_before
                    self.progress.benchmark_fuzzy_matches_before = (
                        gap.fuzzy_matches_before
                    )
                    self.progress.benchmark_missed_before = len(gap.missed_before)
                    self.progress.benchmark_recovered_via_gap_fill = (
                        gap.recovered_via_gap_fill
                    )
                    self.progress.benchmark_still_missing = len(gap.still_missing)
                    self.progress.benchmark_blind_recall = gap.blind_recall
                    self.progress.benchmark_blind_recall_fuzzy = gap.blind_recall_fuzzy
                    self.progress.benchmark_total_recall = gap.total_recall
                    self.progress.benchmark_still_missing_sample = list(
                        gap.still_missing[:50]
                    )

                    # Persist the full still-missing list to the run dir.
                    if self._run_dir:
                        try:
                            (self._run_dir / "benchmark_report.json").write_text(
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
                        loop.run_until_complete(self._schemify.normalize())
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("benchmark final normalize failed: %s", e)

                    self._df = self._schemify.to_dataframe()
                    self._dataset_json = self._build_dataset_json()

                    # Pull a fresh live snapshot so query_count reflects
                    # ALL phases (blind loop + gap-fill), not just the
                    # last on_progress callback fire.
                    try:
                        live = self._schemify.get_live_progress() or {}
                        self.progress.query_count = max(
                            int(live.get("query_count", 0) or 0),
                            self.progress.query_count,
                        )
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("get_live_progress() at finalize failed: %s", e)

                    stats = self._schemify.get_stats()
                    llm_usage = stats.get("llm_usage", {}) or {}
                    self.usage = UsageStats(
                        total_tokens=int(llm_usage.get("total_tokens", 0) or 0),
                        total_cost_usd=float(
                            llm_usage.get("total_cost_usd", 0.0) or 0.0
                        ),
                        queries_run=self.progress.query_count,
                    )

                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = (
                        f"Rediscovery benchmark complete — "
                        f"blind {gap.blind_recall:.0%}, "
                        f"with gap-fill {gap.total_recall:.0%}"
                    )
                    if self._schemify.record_set:
                        self.progress.entity_count = len(
                            self._schemify.record_set.records
                        )
                    self._bump_record_version()

                    try:
                        self._save_run(category=category)
                    except OSError as e:
                        _logger.warning("_save_run after benchmark failed: %s", e)
                finally:
                    loop.close()

            except Exception as e:  # noqa: BLE001
                _logger.exception("rediscovery-benchmark worker failed")
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def can_continue_research(self) -> bool:
        """True iff the in-memory Schemify state is alive enough to keep researching."""
        return bool(
            self._schemify
            and self._schemify.record_set
            and not self.is_running
        )

    @property
    def is_read_only(self) -> bool:
        """True when the loaded dataset has no live Schemify instance
        (e.g. a saved run loaded without an API key). Curation and
        continue-research are unavailable in this state."""
        return self._read_only_reason is not None and self._schemify is None

    @property
    def read_only_reason(self) -> Optional[str]:
        """Human-readable explanation of why the dataset is read-only, or None."""
        return self._read_only_reason if self.is_read_only else None

    def continue_research(
        self,
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
        if not self.can_continue_research():
            return False

        category = self._schemify.record_set.category if self._schemify.record_set else ""
        # Reset completion / error flags but keep entity_count etc.
        self.progress.is_running = True
        self.progress.is_complete = False
        self.progress.error = None
        self.progress.stage = "Continuing research…"
        self.progress.current = 0
        self.progress.total = 0

        def _run() -> None:
            try:
                def _on_progress(stage: str, current: int, total: int) -> None:
                    self.progress.stage = stage
                    self.progress.current = current
                    self.progress.total = total
                    if self._schemify and self._schemify.record_set:
                        snap = None
                        getter = getattr(self._schemify, "get_live_progress", None)
                        if callable(getter):
                            try:
                                snap = getter()
                            except Exception as e:  # noqa: BLE001
                                _logger.warning("get_live_progress() failed: %s", e)
                        if snap is not None:
                            self.progress.entity_count = int(snap.get("entity_count", 0))
                            self.progress.query_count = max(
                                int(snap.get("query_count", 0)), current
                            )
                        else:
                            self.progress.entity_count = len(self._schemify.record_set.records)
                            history_count = len(
                                getattr(self._schemify, "query_history", []) or []
                            )
                            counter = int(
                                getattr(self._schemify, "_query_counter", 0) or 0
                            )
                            self.progress.query_count = max(history_count, counter, current)
                        self._bump_record_version()
                    try:
                        self._snapshot_partial(category=category, throttle=True)
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("snapshot_partial failed: %s", e)

                self._schemify.on_progress(_on_progress)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    self.progress.stage = "Running additional web search queries…"
                    loop.run_until_complete(
                        self._schemify.run_agentic(
                            max_queries=max_queries,
                            concurrency=concurrency,
                            phase_split=phase_split,
                            output_dir=str(self._run_dir) if self._run_dir else None,
                        )
                    )

                    if verify:
                        self.progress.stage = "Verifying attribute values…"
                        loop.run_until_complete(
                            self._schemify.verify_unverified(concurrency=concurrency)
                        )

                    self.progress.stage = "Finalizing dataset…"
                    self._schemify.finalize()

                    self._df = self._schemify.to_dataframe()
                    self._dataset_json = self._build_dataset_json()

                    stats = self._schemify.get_stats()
                    llm_usage = stats.get("llm_usage", {})
                    self.usage = UsageStats(
                        total_tokens=llm_usage.get("total_tokens", 0),
                        total_cost_usd=llm_usage.get("total_cost_usd", 0.0),
                        queries_run=self.progress.query_count,
                    )

                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = "Complete"
                    if self._schemify.record_set:
                        self.progress.entity_count = len(self._schemify.record_set.records)
                    self._bump_record_version()

                    try:
                        self._save_run(category=category)
                    except OSError as e:
                        _logger.warning("_save_run after continue failed: %s", e)
                finally:
                    loop.close()

            except Exception as e:  # noqa: BLE001
                _logger.exception("continue_research worker failed")
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

    # ── On-demand verification ──────────────────────────────

    def count_unverified(self) -> tuple[int, int]:
        """Return ``(entities_with_unverified, total_unverified_values)``.

        An attribute value is "unverified" when it has zero distinct
        citations (``av.evidence.source_count == 0``).
        """
        if not self._schemify or not self._schemify.record_set:
            return (0, 0)
        entities = 0
        values = 0
        for record in self._schemify.record_set.records:
            had = False
            for bucket in (record.attributes, record.additional_attributes):
                for av in bucket.values():
                    if av.evidence.source_count == 0:
                        values += 1
                        had = True
            if had:
                entities += 1
        return (entities, values)

    def start_verification(self, concurrency: int = 12) -> None:
        """Run verification in a background daemon thread."""
        if self.is_running or not self._schemify:
            return
        total_entities = (
            len(self._schemify.record_set.records)
            if self._schemify.record_set
            else 0
        )
        self.progress = ResearchProgress(
            is_running=True,
            stage="Verifying attribute values…",
            entity_count=total_entities,
        )

        def _on_verify_progress(done: int, total: int, label: str) -> None:
            # Called from the verification event loop thread. Mutating the
            # dataclass fields is fine — the UI polls them.
            self.progress.current = done
            self.progress.total = total
            self.progress.stage = (
                f"Verifying {done}/{total}: {label}" if label else
                f"Verifying {done}/{total}…"
            )

        def _run() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self._schemify.verify_unverified(
                            concurrency=concurrency,
                            progress_callback=_on_verify_progress,
                        )
                    )
                    self._df = self._schemify.to_dataframe()
                    self._dataset_json = self._build_dataset_json()
                    self._bump_record_version()
                    # Refresh usage figures.
                    try:
                        stats = self._schemify.get_stats()
                        llm_usage = stats.get("llm_usage", {}) or {}
                        self.usage = UsageStats(
                            total_tokens=int(llm_usage.get("total_tokens", 0) or 0),
                            total_cost_usd=float(
                                llm_usage.get("total_cost_usd", 0.0) or 0.0
                            ),
                            queries_run=self.progress.query_count,
                        )
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("usage refresh after verify failed: %s", e)
                    try:
                        self._snapshot_partial(
                            category=self._schemify.record_set.category
                            if self._schemify.record_set
                            else ""
                        )
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("snapshot after verify failed: %s", e)
                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = "Verification complete"
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
                _logger.exception("verification worker failed")
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ── Candidate seeding (L) ──────────────────────────────────

    def append_guidance(self, note: str) -> bool:
        """Append a free-form user note to the record set's research guidance.

        The text is added on its own paragraph prefixed with ``User note:`` so
        it shows up clearly in every downstream prompt that interpolates
        ``{guidance}``. Returns True if anything was appended.
        """
        text = (note or "").strip()
        if not text or not self._schemify or not self._schemify.record_set:
            return False
        rs = self._schemify.record_set
        prefix = (rs.guidance or "").rstrip()
        suffix = f"User note: {text}"
        rs.guidance = f"{prefix}\n\n{suffix}" if prefix else suffix
        self._dataset_json = self._build_dataset_json()
        try:
            self._snapshot_partial(category=rs.category)
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot_partial failed: %s", e)
        return True

    @staticmethod
    def parse_candidate_file(filename: str, raw: bytes) -> list[str]:
        """Best-effort extraction of candidate names from an uploaded file.

        Supports plain text (one name per line or comma-separated), CSV
        (first column or a column whose header contains ``name``/``label``/
        ``entity``), and JSON (list of strings or list of dicts with one of
        those keys).
        """
        if not raw:
            return []
        name = (filename or "").lower()
        try:
            text = raw.decode("utf-8-sig", errors="replace")
        except Exception:  # noqa: BLE001
            text = raw.decode("latin-1", errors="replace")

        if name.endswith(".json"):
            try:
                data = json.loads(text)
            except Exception:  # noqa: BLE001
                return []
            out: list[str] = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        for key in ("label", "name", "entity", "title"):
                            v = item.get(key)
                            if isinstance(v, str) and v.strip():
                                out.append(v)
                                break
            return out

        if name.endswith(".csv") or name.endswith(".tsv"):
            sep = "\t" if name.endswith(".tsv") else ","
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
            except Exception:  # noqa: BLE001
                return [line.strip() for line in text.splitlines() if line.strip()]
            cols = list(df.columns)
            target = cols[0]
            for c in cols:
                lc = str(c).lower()
                if any(k in lc for k in ("label", "name", "entity")):
                    target = c
                    break
            return [str(v).strip() for v in df[target].dropna().tolist() if str(v).strip()]

        # Plain text fallback: split on commas or newlines.
        return [
            tok.strip()
            for tok in text.replace(",", "\n").splitlines()
            if tok.strip()
        ]

    def add_candidate_entities(self, names: list[str]) -> int:
        """Add user-supplied entity labels as blank seed records.

        Returns the number of records actually added (skips duplicates).
        """
        if not self._schemify or not self._schemify.record_set:
            return 0
        from intelligence_toolkit.schemify.models import Record

        added = 0
        existing = {
            (r.label or "").strip().casefold()
            for r in self._schemify.record_set.records
        }
        for raw in names:
            name = (raw or "").strip()
            if not name or name.casefold() in existing:
                continue
            rec = Record(label=name.upper())
            ok, _ = self._schemify.record_set.add_record(rec, use_fuzzy=False)
            if ok:
                added += 1
                existing.add(name.casefold())
        if added:
            self._df = self._schemify.to_dataframe()
            self._dataset_json = self._build_dataset_json()
            self._bump_record_version()
            try:
                self._snapshot_partial(
                    category=self._schemify.record_set.category
                )
            except Exception as e:  # noqa: BLE001
                _logger.warning("snapshot_partial failed: %s", e)
        return added

    # ── Exclusions ─────────────────────────────────────────────

    @property
    def exclusions(self) -> list[dict]:
        """Current user-supplied exclusion rules (label-based or attribute predicates)."""
        if not self._schemify or not self._schemify.record_set:
            return []
        return list(self._schemify.record_set.user_exclusions or [])

    def add_label_exclusion(
        self, label: str, reason: str = "", remove_existing: bool = True
    ) -> tuple[bool, int]:
        """Exclude a single named entity. See add_attribute_exclusion for predicate rules."""
        if not self._schemify or not self._schemify.record_set:
            return (False, 0)
        lbl = (label or "").strip()
        if not lbl:
            return (False, 0)
        rule = {"label": lbl, "reason": (reason or "").strip()}
        return self._upsert_exclusion(rule, remove_existing=remove_existing)

    # Backwards-compatible alias used by older UI code / scripts.
    def add_exclusion(
        self, label: str, reason: str = "", remove_existing: bool = True
    ) -> tuple[bool, int]:
        return self.add_label_exclusion(label, reason, remove_existing)

    def add_attribute_exclusion(
        self,
        attribute: str,
        operator: str,
        values: list[str] | None = None,
        reason: str = "",
        remove_existing: bool = True,
    ) -> tuple[bool, int]:
        """Exclude every entity that satisfies the given attribute predicate.

        Supported operators: ``missing``, ``equals``, ``in``, ``contains``, ``regex``.
        """
        if not self._schemify or not self._schemify.record_set:
            return (False, 0)
        attr = (attribute or "").strip()
        op = (operator or "equals").strip().lower()
        if not attr or op not in {"missing", "equals", "in", "contains", "regex"}:
            return (False, 0)
        vals = [str(v).strip() for v in (values or []) if str(v).strip()]
        if op != "missing" and not vals:
            return (False, 0)
        rule = {
            "attribute": attr,
            "operator": op,
            "values": vals,
            "reason": (reason or "").strip(),
        }
        return self._upsert_exclusion(rule, remove_existing=remove_existing)

    def remove_exclusion(self, identifier: str | dict) -> bool:
        """Remove an exclusion rule. ``identifier`` may be a label string
        (matches label rules) or a rule dict (compared by attribute+operator+values).
        """
        if not self._schemify or not self._schemify.record_set:
            return False
        rs = self._schemify.record_set
        before = len(rs.user_exclusions or [])
        rs.user_exclusions = [
            e for e in (rs.user_exclusions or [])
            if not self._rule_matches_identifier(e, identifier)
        ]
        if len(rs.user_exclusions) == before:
            return False
        self._dataset_json = self._build_dataset_json()
        try:
            self._snapshot_partial(category=rs.category)
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot_partial failed: %s", e)
        return True

    def _upsert_exclusion(
        self, rule: dict, remove_existing: bool = True
    ) -> tuple[bool, int]:
        rs = self._schemify.record_set
        rs.user_exclusions = list(rs.user_exclusions or [])
        # Dedupe — same kind+target overwrites the previous reason.
        for i, existing in enumerate(rs.user_exclusions):
            if self._rules_equivalent(existing, rule):
                rs.user_exclusions[i] = rule
                break
        else:
            rs.user_exclusions.append(rule)

        removed = self._remove_records_matching_rule(rule) if remove_existing else 0

        self._df = self._schemify.to_dataframe()
        self._dataset_json = self._build_dataset_json()
        try:
            self._snapshot_partial(category=rs.category)
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot_partial failed: %s", e)
        return (True, removed)

    @staticmethod
    def _rules_equivalent(a: dict, b: dict) -> bool:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if bool(a.get("attribute")) != bool(b.get("attribute")):
            return False
        if a.get("attribute"):
            return (
                (a.get("attribute") or "").strip().casefold()
                == (b.get("attribute") or "").strip().casefold()
                and (a.get("operator") or "equals").lower()
                == (b.get("operator") or "equals").lower()
                and [str(v).strip().casefold() for v in (a.get("values") or [])]
                == [str(v).strip().casefold() for v in (b.get("values") or [])]
            )
        return (
            (a.get("label") or "").strip().casefold()
            == (b.get("label") or "").strip().casefold()
        )

    @classmethod
    def _rule_matches_identifier(cls, rule: dict, identifier) -> bool:
        if isinstance(identifier, dict):
            return cls._rules_equivalent(rule, identifier)
        ident = (identifier or "").strip().casefold()
        if not ident:
            return False
        if rule.get("attribute"):
            return False
        return (rule.get("label") or "").strip().casefold() == ident

    def _remove_records_matching_rule(self, rule: dict) -> int:
        if not self._schemify or not self._schemify.record_set:
            return 0
        from intelligence_toolkit.schemify.resolution import record_matches_rule

        rs = self._schemify.record_set
        keep = []
        removed = 0
        for r in rs.records:
            if record_matches_rule(r, rule):
                removed += 1
            else:
                keep.append(r)
        if removed:
            rs.records = keep
            rs.update_schema_frequencies()
        return removed

    def _remove_records_matching(self, label: str) -> int:
        """Legacy helper kept for backward compatibility."""
        return self._remove_records_matching_rule({"label": label})

    # ── Schema editing (K) ─────────────────────────────────────

    def rename_attribute(self, old_name: str, new_name: str) -> int:
        """Rename a schema attribute and update all records. Returns affected records."""
        if not self._schemify or not self._schemify.record_set:
            return 0
        old = (old_name or "").strip()
        new = (new_name or "").strip()
        if not old or not new or old == new:
            return 0
        rs = self._schemify.record_set
        # Schema attributes
        for sa in rs.schema_attributes:
            if sa.name == old:
                sa.name = new
                break
        affected = 0
        for record in rs.records:
            for bucket in (record.attributes, record.additional_attributes):
                if old in bucket:
                    bucket[new] = bucket.pop(old)
                    affected += 1
        rs.update_schema_frequencies()
        self._df = self._schemify.to_dataframe()
        self._dataset_json = self._build_dataset_json()
        try:
            self._snapshot_partial(category=rs.category)
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot_partial failed: %s", e)
        return affected

    def remove_attribute(self, name: str) -> int:
        """Remove a schema attribute from the schema and all records."""
        if not self._schemify or not self._schemify.record_set:
            return 0
        target = (name or "").strip()
        if not target:
            return 0
        rs = self._schemify.record_set
        rs.schema_attributes = [
            sa for sa in rs.schema_attributes if sa.name != target
        ]
        affected = 0
        for record in rs.records:
            for bucket in (record.attributes, record.additional_attributes):
                if target in bucket:
                    bucket.pop(target, None)
                    affected += 1
        rs.update_schema_frequencies()
        self._df = self._schemify.to_dataframe()
        self._dataset_json = self._build_dataset_json()
        try:
            self._snapshot_partial(category=rs.category)
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot_partial failed: %s", e)
        return affected

    # ── Normalization on demand (J) ────────────────────────────

    def start_normalize(self, attributes: Optional[list[str]] = None) -> None:
        """Run schemify.normalize in a background thread for the given attributes."""
        if self.is_running or not self._schemify:
            return
        target = (
            ", ".join(attributes) if attributes else "all attributes"
        )
        self.progress = ResearchProgress(
            is_running=True,
            stage=f"Normalizing {target}…",
            entity_count=len(self._schemify.record_set.records)
            if self._schemify.record_set
            else 0,
        )

        def _on_norm_progress(done: int, total: int, attr_name: str) -> None:
            self.progress.current = done
            self.progress.total = total
            if attr_name:
                self.progress.stage = (
                    f"Normalizing {attr_name} ({done}/{total})"
                )
            else:
                self.progress.stage = f"Normalizing ({done}/{total})"

        def _run() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Pass progress_callback when supported; otherwise fall
                    # back so older Schemify still works.
                    try:
                        coro = self._schemify.normalize(
                            attributes=attributes,
                            progress_callback=_on_norm_progress,
                        )
                    except TypeError:
                        coro = self._schemify.normalize(attributes=attributes)
                    loop.run_until_complete(coro)
                    self._df = self._schemify.to_dataframe()
                    self._dataset_json = self._build_dataset_json()
                    self._bump_record_version()
                    try:
                        self._snapshot_partial(
                            category=self._schemify.record_set.category
                        )
                    except Exception as e:  # noqa: BLE001
                        _logger.warning("snapshot after normalize failed: %s", e)
                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = "Normalization complete"
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
                _logger.exception("normalize worker failed")
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ── Harmful content scan (I) ───────────────────────────────

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
        self,
        prompt_template: Optional[str] = None,
        *,
        concurrency: int = _SAFETY_SCAN_CONCURRENCY,
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
        if not self._schemify or not self._schemify.record_set:
            return []
        try:
            from intelligence_toolkit.AI.client import OpenAIClient
            from intelligence_toolkit.AI.defaults import DEFAULT_TEMPERATURE
        except ImportError as e:
            _logger.warning("safety-scan: AI client import failed: %s", e)
            return []

        template = (prompt_template or self.DEFAULT_SAFETY_PROMPT).strip()
        if "{record}" not in template:
            template = template + "\n\nRECORD:\n{record}\n"

        client = OpenAIClient()
        records = list(self._schemify.record_set.records)

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

        def _scan_one(item: tuple[int, str, dict[str, str]]) -> Optional[dict]:
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

    def remove_record_by_label(self, label: str) -> bool:
        """Drop a single record by exact label (case-insensitive). Returns True if removed."""
        if not self._schemify or not self._schemify.record_set:
            return False
        rs = self._schemify.record_set
        target = (label or "").strip().casefold()
        if not target:
            return False
        keep = []
        removed = False
        for r in rs.records:
            if (r.label or "").strip().casefold() == target and not removed:
                removed = True
                continue
            keep.append(r)
        if removed:
            rs.records = keep
            rs.update_schema_frequencies()
            self._df = self._schemify.to_dataframe()
            self._dataset_json = self._build_dataset_json()
            try:
                self._snapshot_partial(category=rs.category)
            except Exception:  # noqa: BLE001
                pass
        return removed

    # ── Alias / merge curation (M) ─────────────────────────────

    def list_alias_groups(self, only_with_aliases: bool = False) -> list[dict]:
        """Return every record's canonical label + its alias group.

        Each entry is ``{"label", "aliases", "alias_counts", "total_count"}``.
        Sorted by ``total_count`` desc so the biggest merged groups (the
        most likely culprits for over-aggressive merging) appear first.
        """
        if not self._schemify or not self._schemify.record_set:
            return []
        rs = self._schemify.record_set
        groups: list[dict] = []
        for r in rs.records:
            aliases = list(r.aliases or [])
            if only_with_aliases and not aliases:
                continue
            counts = dict(r.alias_counts or {})
            groups.append(
                {
                    "label": r.label,
                    "aliases": aliases,
                    "alias_counts": counts,
                    "total_count": sum(counts.values()) if counts else 1,
                }
            )
        groups.sort(key=lambda g: (-g["total_count"], g["label"]))
        return groups

    @property
    def do_not_merge_pairs(self) -> list[list[str]]:
        """User-asserted pairs that must never be auto-merged again."""
        if not self._schemify or not self._schemify.record_set:
            return []
        rs = self._schemify.record_set
        pairs = getattr(rs, "do_not_merge", None) or set()
        return [sorted(list(p)) for p in pairs]

    def _record_do_not_merge(self, a: str, b: str) -> None:
        rs = self._schemify.record_set
        if not hasattr(rs, "do_not_merge") or rs.do_not_merge is None:
            rs.do_not_merge = set()
        a_norm = (a or "").strip()
        b_norm = (b or "").strip()
        if a_norm and b_norm and a_norm.casefold() != b_norm.casefold():
            rs.do_not_merge.add(frozenset((a_norm, b_norm)))

    def _find_record(self, label: str):
        """Find a record by exact label or by alias (case-insensitive)."""
        if not self._schemify or not self._schemify.record_set:
            return None
        target = (label or "").strip().casefold()
        if not target:
            return None
        for r in self._schemify.record_set.records:
            if (r.label or "").strip().casefold() == target:
                return r
            for a in r.aliases or []:
                if (a or "").strip().casefold() == target:
                    return r
        return None

    def rename_record_canonical(self, current_label: str, new_canonical: str) -> bool:
        """Promote ``new_canonical`` to be the canonical label of the record
        currently identified by ``current_label`` (or one of its aliases).

        - If ``new_canonical`` already exists as an alias, it is promoted and
          the previous label is demoted to an alias.
        - If it is a brand-new string, it is added (with count 1) before
          promotion.
        - A ``manual_canonical = True`` marker is set on the record so
          subsequent alias additions/merges do not silently rename it.
        """
        rec = self._find_record(current_label)
        if rec is None:
            return False
        new_label = (new_canonical or "").strip()
        if not new_label:
            return False
        if new_label.upper() == (rec.label or "").upper():
            rec.manual_canonical = True
            self._post_curation_refresh()
            return True
        # Ensure the new label is in alias_counts (with at least count 1).
        existing_key = None
        for key in rec.alias_counts:
            if key.lower() == new_label.lower():
                existing_key = key
                break
        if existing_key is None:
            rec.alias_counts[new_label] = 1
        else:
            new_label = existing_key  # preserve original casing of the alias
        # Demote current label to alias.
        if rec.label and rec.label not in rec.aliases:
            rec.aliases.append(rec.label)
        if rec.label and rec.label not in rec.alias_counts:
            rec.alias_counts[rec.label] = 1
        rec.label = new_label
        rec.aliases = [
            name for name in rec.alias_counts.keys()
            if name.upper() != rec.label.upper()
        ]
        rec.manual_canonical = True
        self._post_curation_refresh()
        return True

    def remove_alias(self, label: str, alias: str) -> bool:
        """Drop ``alias`` from the record's alias list and counts. Does not
        re-introduce it as a separate record."""
        rec = self._find_record(label)
        if rec is None:
            return False
        target = (alias or "").strip()
        if not target:
            return False
        # Don't allow deleting the canonical via this path.
        if target.upper() == (rec.label or "").upper():
            return False
        existing_key = None
        for key in rec.alias_counts:
            if key.lower() == target.lower():
                existing_key = key
                break
        changed = False
        if existing_key is not None:
            rec.alias_counts.pop(existing_key, None)
            changed = True
        rec.aliases = [
            a for a in rec.aliases
            if (a or "").lower() != target.lower()
        ]
        if changed:
            self._post_curation_refresh()
        return changed

    def split_alias(self, label: str, alias: str) -> bool:
        """Extract ``alias`` from the given record into a new standalone
        record (preserving its alias count) and register a do-not-merge
        constraint between the two so subsequent fuzzy passes won't
        re-merge them."""
        if not self._schemify or not self._schemify.record_set:
            return False
        from intelligence_toolkit.schemify.models import Record

        rec = self._find_record(label)
        if rec is None:
            return False
        target = (alias or "").strip()
        if not target:
            return False
        if target.upper() == (rec.label or "").upper():
            return False

        existing_key = None
        for key in rec.alias_counts:
            if key.lower() == target.lower():
                existing_key = key
                break
        if existing_key is None:
            return False

        count = int(rec.alias_counts.pop(existing_key, 1) or 1)
        rec.aliases = [
            a for a in rec.aliases
            if (a or "").lower() != target.lower()
        ]

        # Create a fresh record from the extracted alias. Use upper-case to
        # match the rest of the dataset convention.
        new_rec = Record(label=existing_key.upper())
        new_rec.alias_counts[new_rec.label] = count
        # Mark as user-pinned so it won't be silently renamed if it picks up
        # a more frequent alias later.
        new_rec.manual_canonical = True
        self._schemify.record_set.records.append(new_rec)

        # Register the do-not-merge constraint (both directions of label).
        self._record_do_not_merge(rec.label, new_rec.label)

        self._post_curation_refresh()
        return True

    def merge_records(self, primary_label: str, other_label: str) -> bool:
        """Merge the record identified by ``other_label`` into the one
        identified by ``primary_label``. The primary's canonical label is
        preserved (and pinned) so the merge never flips canonicals on the
        user."""
        if not self._schemify or not self._schemify.record_set:
            return False
        primary = self._find_record(primary_label)
        other = self._find_record(other_label)
        if primary is None or other is None or primary is other:
            return False
        # Pin canonical on the primary so the absorbed counts can't promote
        # an alias from ``other`` to canonical.
        primary.manual_canonical = True
        primary.merge_from(other)
        try:
            self._schemify.record_set.records.remove(other)
        except ValueError:
            pass
        # Drop any stale do-not-merge constraint between these two — the
        # user has now explicitly opted in.
        rs = self._schemify.record_set
        if getattr(rs, "do_not_merge", None):
            rs.do_not_merge = {
                p for p in rs.do_not_merge
                if not (primary.label in p and other.label in p)
            }
        self._post_curation_refresh()
        return True

    def unpin_canonical(self, label: str) -> bool:
        """Allow automatic canonical re-selection on this record."""
        rec = self._find_record(label)
        if rec is None:
            return False
        if getattr(rec, "manual_canonical", False):
            rec.manual_canonical = False
            # Recompute now so the UI reflects the change.
            rec._update_canonical_label()
            self._post_curation_refresh()
            return True
        return False

    def _post_curation_refresh(self) -> None:
        """Refresh derived state after any merge/split/rename edit."""
        if not self._schemify or not self._schemify.record_set:
            return
        rs = self._schemify.record_set
        try:
            rs.update_schema_frequencies()
        except Exception as e:  # noqa: BLE001
            _logger.warning("update_schema_frequencies failed: %s", e)
        try:
            self._df = self._schemify.to_dataframe()
        except Exception as e:  # noqa: BLE001
            _logger.warning("to_dataframe after curation failed: %s", e)
        self._dataset_json = self._build_dataset_json()
        self._bump_record_version()
        try:
            self._snapshot_partial(category=rs.category)
        except Exception as e:  # noqa: BLE001
            _logger.warning("snapshot after curation failed: %s", e)

    # ── AI-suggested alias groupings ───────────────────────────

    DEFAULT_ALIAS_SUGGEST_PROMPT = (
        "You are curating an entity dataset about: {category}.\n\n"
        "Below are entity labels currently treated as DISTINCT records. "
        "Identify groups of two or more labels in this list that refer to "
        "the SAME real-world entity (e.g. acronym + full name, vendor "
        "renaming, common typo). Be conservative: do NOT group entities "
        "that merely belong to the same category, are made by the same "
        "organisation, or share a generic word.\n\n"
        "Reply with a JSON array. Each item must be an object with:\n"
        "  - \"primary\": the label that should be the canonical name "
        "(prefer the most specific / official name; avoid category-like "
        "phrases such as 'TOOLS', 'PLATFORMS', 'ARCHIVES').\n"
        "  - \"members\": list of OTHER labels that should be merged into "
        "the primary as aliases. Must be a subset of the input labels.\n"
        "  - \"reason\": one short sentence justifying the merge.\n"
        "If no groupings are warranted, reply with an empty array: [].\n\n"
        "Labels:\n{labels}\n"
    )

    def suggest_alias_groups(
        self,
        max_groups: int = 25,
        prompt_template: Optional[str] = None,
        *,
        chunk_size: int = _ALIAS_SUGGEST_CHUNK_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[dict]:
        """Ask the LLM to propose record-level merges across the current
        dataset.

        Labels are partitioned into chunks of ``chunk_size`` and one LLM
        call is dispatched per chunk so the whole dataset is considered
        instead of being silently truncated. Suggestions are de-duplicated
        across chunks (by primary+members signature).

        Returns a list of ``{"primary": str, "members": list[str],
        "reason": str}`` dicts. Members and primary are validated against
        the current set of canonical labels so applying a suggestion is
        always well-defined.
        """
        if not self._schemify or not self._schemify.record_set:
            return []
        try:
            from intelligence_toolkit.AI.client import OpenAIClient
            from intelligence_toolkit.AI.defaults import DEFAULT_TEMPERATURE
        except ImportError as e:
            _logger.warning("alias-suggest: AI client import failed: %s", e)
            return []

        rs = self._schemify.record_set
        labels = [r.label for r in rs.records if (r.label or "").strip()]
        if len(labels) < 2:
            return []

        # Build deterministic chunks. For datasets larger than chunk_size
        # we shuffle the chunk boundaries with a stable hash so labels
        # that *should* group don't end up always in different chunks
        # (e.g. alphabetical-by-default would split "IBM" and "International
        # Business Machines").
        chunk = max(2, int(chunk_size))
        if len(labels) > chunk:
            # Stable but spread-out ordering: round-robin into ``num_chunks``
            # buckets by index. Two passes would be required to catch all
            # cross-chunk merges, but a single pass already finds within-chunk
            # ones reliably.
            num_chunks = (len(labels) + chunk - 1) // chunk
            buckets: list[list[str]] = [[] for _ in range(num_chunks)]
            for i, lbl in enumerate(labels):
                buckets[i % num_chunks].append(lbl)
            chunks = buckets
        else:
            chunks = [list(labels)]

        template = (prompt_template or self.DEFAULT_ALIAS_SUGGEST_PROMPT).strip()
        client = OpenAIClient()
        label_lookup = {lbl.casefold(): lbl for lbl in labels}
        cleaned: list[dict] = []
        seen_keys: set[tuple[str, tuple[str, ...]]] = set()

        for idx, chunk_labels in enumerate(chunks):
            if progress_callback:
                try:
                    progress_callback(idx, len(chunks))
                except Exception as e:  # noqa: BLE001
                    _logger.warning("alias-suggest progress raised: %s", e)
            labels_block = "\n".join(f"- {lbl}" for lbl in chunk_labels)
            prompt = template.format(
                category=rs.category or "(unspecified)",
                labels=labels_block,
            )
            try:
                raw = client.generate_chat(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=DEFAULT_TEMPERATURE,
                )
            except Exception as e:  # noqa: BLE001
                _logger.warning("alias-suggest: chunk %d/%d failed: %s",
                                idx + 1, len(chunks), e)
                continue

            for s in _parse_alias_suggestions(raw or ""):
                primary_raw = (s.get("primary") or "").strip()
                primary = label_lookup.get(primary_raw.casefold())
                members_raw = s.get("members") or []
                members: list[str] = []
                seen_m = set()
                for m in members_raw:
                    if not isinstance(m, str):
                        continue
                    resolved = label_lookup.get(m.strip().casefold())
                    if (
                        resolved
                        and resolved != primary
                        and resolved.casefold() not in seen_m
                    ):
                        members.append(resolved)
                        seen_m.add(resolved.casefold())
                if not (primary and members):
                    continue
                key = (
                    primary.casefold(),
                    tuple(sorted(m.casefold() for m in members)),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                cleaned.append(
                    {
                        "primary": primary,
                        "members": members,
                        "reason": (s.get("reason") or "").strip(),
                    }
                )
                if len(cleaned) >= max_groups:
                    break
            if len(cleaned) >= max_groups:
                break

        if progress_callback:
            try:
                progress_callback(len(chunks), len(chunks))
            except Exception as e:  # noqa: BLE001
                _logger.warning("alias-suggest progress raised: %s", e)
        return cleaned

    def apply_alias_suggestion(self, primary: str, members: list[str]) -> int:
        """Merge each ``members[i]`` into ``primary``. Returns merges done."""
        applied = 0
        for m in members or []:
            if self.merge_records(primary, m):
                applied += 1
        return applied

    # ── Merge-quality audit ────────────────────────────────────

    def audit_merge_quality(
        self,
        *,
        api_key: Optional[str] = None,
        budget: float = 4.0,
        model: Optional[str] = None,
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
        from intelligence_toolkit.schemify import merge_audit as _ma
        from intelligence_toolkit.schemify.llm import LLMClient
        from intelligence_toolkit.schemify.models import SchemifyConfig

        if not self._schemify or not self._schemify.record_set:
            return {
                "results": [], "flagged": [],
                "candidates": 0, "total_records": 0,
            }
        rs = self._schemify.record_set

        # Reuse the active llm if we have one; otherwise build a fresh
        # client from the supplied key. We need an api_key one way or
        # the other — the audit makes one LLM call per candidate.
        llm = getattr(self._schemify, "llm", None)
        if llm is None:
            if not api_key:
                raise ValueError("audit_merge_quality requires api_key when no llm is attached")
            cfg_kwargs = {"api_key": api_key, "max_budget": float(budget)}
            if model:
                cfg_kwargs["model"] = model
            llm = LLMClient(SchemifyConfig(**cfg_kwargs))

        async def _run() -> list:
            return await _ma.audit_records(
                rs, llm,
                concurrency=concurrency,
                progress_cb=progress_cb,
            )
        results = asyncio.run(_run())
        flagged = _ma.flagged_results(
            results, confidence_threshold=confidence_threshold
        )
        return {
            "total_records": len(rs.records),
            "candidates": len(results),
            "flagged": [r.to_dict() for r in flagged],
            "results": [r.to_dict() for r in results],
        }

    def apply_audit_split(self, label: str, audit_entry: dict) -> int:
        """Apply a single split proposal (an entry from the audit's
        ``flagged`` list) to the record carrying ``label``. Returns the
        number of new records created (0 if no change).
        """
        from intelligence_toolkit.schemify import merge_audit as _ma

        if not self._schemify or not self._schemify.record_set:
            return 0
        rs = self._schemify.record_set
        target = None
        target_idx = -1
        for i, r in enumerate(rs.records):
            if (getattr(r, "label", "") or "").strip().upper() == (label or "").strip().upper():
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
        rs.records[target_idx:target_idx + 1] = new_records
        # Lock the splits as do_not_merge so a future pass won't reunite.
        if not hasattr(rs, "do_not_merge") or rs.do_not_merge is None:
            rs.do_not_merge = set()
        for a, b in dnm_pairs:
            if a and b and a != b:
                rs.do_not_merge.add(frozenset({a, b}))
        self._post_curation_refresh()
        return len(new_records)




    def _build_dataset_json(self) -> Optional[dict]:
        """Serialize the current record set to a JSON-able dict.

        ``finalize_normalization`` (called from ``_snapshot_partial`` and
        ``Schemify.finalize``) is responsible for pruning empty schema
        attributes and placeholder values. This builder just serializes
        the current state — no extra filtering here, so the dashboard,
        CSV, and on-screen dataframe all see the same columns.
        """
        rs = getattr(self._schemify, "record_set", None) if self._schemify else None
        if not rs:
            return None
        return {
            "category": rs.category,
            "guidance": rs.guidance,
            "schema_attributes": [
                {
                    "name": a.name,
                    "description": getattr(a, "description", ""),
                    "is_closed_set": getattr(a, "is_closed_set", False),
                    "is_multi_valued": getattr(a, "is_multi_valued", False),
                }
                for a in rs.schema_attributes
            ],
            "records": [
                r.to_dict() for r in rs.records if hasattr(r, "to_dict")
            ],
        }

    def _snapshot_partial(self, category: str, *, throttle: bool = False) -> None:
        """Write the in-progress dataset to ``<run_dir>/data.json``.

        Pass ``throttle=True`` from the per-query progress callback to
        rate-limit writes to ``_SNAPSHOT_MIN_INTERVAL_SECONDS``. User-
        initiated mutations (curation, exclusions, etc.) use the default
        and persist immediately.
        """
        if self._run_dir is None:
            return
        if throttle:
            now = time.monotonic()
            if (now - self._last_snapshot_time) < _SNAPSHOT_MIN_INTERVAL_SECONDS:
                self._snapshot_pending = True
                return
            self._last_snapshot_time = now
            self._snapshot_pending = False
        else:
            self._last_snapshot_time = time.monotonic()
            self._snapshot_pending = False
        # Apply deterministic finalization (ALL-CAPS labels, fold
        # "Also known as" into aliases, lift units into attribute names)
        # so the live preview and on-disk snapshot match the final state.
        try:
            if self._schemify and self._schemify.record_set:
                self._schemify.resolution.finalize_normalization(
                    self._schemify.record_set
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning("finalize_normalization failed during snapshot: %s", e)
        data = self._build_dataset_json()
        if not data:
            return
        try:
            (self._run_dir / "data.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            ts = self._run_dir.name.split("_", 1)[0]
            meta = {
                "category": category,
                "timestamp": ts,
                "entity_count": len(data.get("records", [])),
                "total_tokens": self.usage.total_tokens,
                "total_cost_usd": self.usage.total_cost_usd,
                "queries_run": self.usage.queries_run,
                "in_progress": True,
            }
            (self._run_dir / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except OSError as e:
            _logger.warning("snapshot write failed: %s", e)

    def _save_run(self, category: str) -> Optional[Path]:
        """Write the completed dataset to the run directory so it can be resumed."""
        if not self._dataset_json:
            return None
        if self._run_dir is None:
            _RUNS_DIR.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
            ts = time.strftime("%Y%m%d-%H%M%S")
            self._run_dir = _RUNS_DIR / f"{ts}_{safe}"
            self._run_dir.mkdir(parents=True, exist_ok=True)
        else:
            ts = self._run_dir.name.split("_", 1)[0]
        run_dir = self._run_dir
        data_path = run_dir / "data.json"
        data_path.write_text(
            json.dumps(self._dataset_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        meta = {
            "category": category,
            "timestamp": ts,
            "entity_count": len(self._dataset_json.get("records", [])),
            "total_tokens": self.usage.total_tokens,
            "total_cost_usd": self.usage.total_cost_usd,
            "queries_run": self.usage.queries_run,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return data_path

    @staticmethod
    def list_saved_runs() -> list[dict]:
        """Return saved runs, newest first."""
        if not _RUNS_DIR.exists():
            return []
        runs: list[dict] = []
        for run_dir in _RUNS_DIR.iterdir():
            if not run_dir.is_dir():
                continue
            data_path = run_dir / "data.json"
            if not data_path.exists():
                continue
            meta_path = run_dir / "meta.json"
            meta: dict = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    meta = {}
            runs.append(
                {
                    "name": run_dir.name,
                    "path": str(data_path),
                    "category": meta.get("category", ""),
                    "timestamp": meta.get("timestamp", run_dir.name),
                    "entity_count": meta.get("entity_count", 0),
                    "total_cost_usd": meta.get("total_cost_usd", 0.0),
                    "queries_run": meta.get("queries_run", 0),
                }
            )
        runs.sort(key=lambda r: r["name"], reverse=True)
        return runs

    def load_saved_run(
        self,
        data_path: str | Path,
        api_key: Optional[str] = None,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
    ) -> None:
        """Load a dataset previously written by ``_save_run``.

        When ``api_key`` is supplied, also rehydrates a live ``Schemify``
        instance pointed at the loaded record set so that the user can
        continue research, add seed entities and curate aliases on the
        loaded run. Without an api key the dataset is loaded read-only.
        """
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.load_dataset(data)

        # Point the run dir at the loaded run so further snapshots / saves
        # accumulate alongside the original data.json + meta.json.
        try:
            self._run_dir = Path(data_path).parent
        except (TypeError, ValueError) as e:
            _logger.warning("could not resolve run_dir from %s: %s", data_path, e)
            self._run_dir = None

        self._read_only_reason = None
        if api_key:
            try:
                self._rehydrate_schemify(
                    data, api_key=api_key, model=model, budget=budget
                )
            except Exception as e:  # noqa: BLE001
                _logger.exception("rehydrate_schemify failed")
                self._schemify = None
                self._read_only_reason = (
                    f"Failed to rehydrate live research state ({e}). "
                    "Reload with a valid API key to continue research."
                )
        else:
            self._schemify = None
            self._read_only_reason = (
                "Loaded without an API key — dataset is read-only. "
                "Provide an API key in Settings and reload this run to "
                "continue research or curate aliases."
            )
        self._bump_record_version()

        entity_count = len(data.get("records", []))
        self.progress = ResearchProgress(
            stage="Loaded from saved run",
            entity_count=entity_count,
            query_count=int((self._dataset_json or {}).get("query_count", 0) or 0),
            is_complete=True,
        )

    def _rehydrate_schemify(
        self,
        data: dict,
        *,
        api_key: str,
        model: str = config.DEFAULT_MODEL,
        budget: float = 10.0,
    ) -> None:
        """Build a fresh Schemify and seed it with ``data`` so the loaded
        run can be continued, seeded and curated.
        """
        from intelligence_toolkit.schemify import Schemify  # noqa: PLC0415
        from intelligence_toolkit.schemify.models import (  # noqa: PLC0415
            RecordSet,
            SchemifyConfig,
        )

        cfg = SchemifyConfig(
            api_key=api_key,
            search_model=model,
            completion_model=model,
            max_budget=budget,
            cache_enabled=True,
        )
        sch = Schemify(cfg)
        sch.record_set = RecordSet.from_dict(data)
        self._schemify = sch

    def load_dataset(self, data: dict) -> None:
        """Load a previously saved dataset JSON into this object."""
        self._dataset_json = data
        records = data.get("records", [])
        attrs = [a.get("name", "") for a in data.get("schema_attributes", [])]
        rows = []
        for r in records:
            row: dict = {"name": r.get("label", r.get("name", ""))}
            for attr in attrs:
                raw = r.get("attributes", {}).get(attr, {})
                vals = raw.get("values", []) if isinstance(raw, dict) else []
                row[attr] = "; ".join(v.get("value", "") for v in vals if isinstance(v, dict))
            rows.append(row)
        self._df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Export helpers ─────────────────────────────────────────

    def get_dataset_bytes_json(self) -> bytes:
        if self._dataset_json:
            return json.dumps(self._dataset_json, indent=2, ensure_ascii=False).encode()
        return b"{}"

    def get_dataset_bytes_csv(self) -> bytes:
        if self._df is not None and not self._df.empty:
            return self._df.to_csv(index=False).encode()
        return b""

    def build_dashboard_zip(
        self,
        title: str,
        subtitle: str,
        dataset_label: str,
        primary_color: str,
        accent_color: str,
        logo_bytes: Optional[bytes] = None,
        logo_filename: Optional[str] = None,
        favicon_bytes: Optional[bytes] = None,
        favicon_filename: Optional[str] = None,
        views: Optional[list[str]] = None,
        secondary_accent_color: Optional[str] = None,
        colors: Optional[dict[str, str]] = None,
        footer: Optional[str] = None,
    ) -> bytes:
        """Bundle dashboard.html + theme + data into a downloadable ZIP.

        ``views`` optionally restricts the dashboard's tab bar. Accepted
        values are ``"table"``, ``"cards"``, ``"network"``. ``None`` (the
        default) or an empty list keeps all three.

        ``colors`` overrides any of the 12 theme color slots: primary,
        accent, accentSoft, accent2, warn, highlight, success, bg, surface,
        text, muted, border. Values not provided fall back to defaults
        derived from ``primary_color`` / ``accent_color`` /
        ``secondary_accent_color``.
        """
        dashboard_src = (
            Path(__file__).resolve().parents[1] / "schemify" / "dashboard" / "dashboard.html"
        )

        normalised_views: list[str] = []
        if views:
            allowed = {"table", "cards", "network"}
            for v in views:
                key = str(v).strip().lower()
                if key in allowed and key not in normalised_views:
                    normalised_views.append(key)

        default_colors = {
            "primary": primary_color,
            "accent": accent_color,
            "accentSoft": _lighten(accent_color),
            "accent2": secondary_accent_color or accent_color,
            "warn": "#E85D4A",
            "highlight": "#F0A830",
            "success": "#4CAF50",
            "bg": "#F5F7FA",
            "surface": "#FFFFFF",
            "text": "#1A2A3A",
            "muted": "#6B7C8D",
            "border": "#DDE3EA",
        }
        if colors:
            for k, v in colors.items():
                if isinstance(v, str) and v:
                    default_colors[k] = v

        theme = {
            "name": "custom",
            "title": title,
            "subtitle": subtitle,
            "datasetLabel": dataset_label,
            "logo": logo_filename or "",
            "logoAlt": subtitle,
            "favicon": favicon_filename or "",
            "footer": footer or "Built with Intelligence Toolkit.",
            "views": normalised_views,
            "colors": default_colors,
        }

        theme_js = "window.SCHEMIFY_THEME = " + json.dumps(theme, indent=2) + ";\n"
        data_js = (
            "const DASHBOARD_DATA = "
            + json.dumps(self._dataset_json or {}, ensure_ascii=False)
            + ";\n"
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if dashboard_src.exists():
                zf.writestr("dashboard/dashboard.html", dashboard_src.read_bytes())
            zf.writestr("dashboard/theme.js", theme_js.encode())
            zf.writestr("dashboard/dashboard_data.js", data_js.encode())
            if logo_bytes and logo_filename:
                zf.writestr(f"dashboard/{logo_filename}", logo_bytes)
            if favicon_bytes and favicon_filename:
                zf.writestr(f"dashboard/{favicon_filename}", favicon_bytes)
        buf.seek(0)
        return buf.read()


def _lighten(hex_color: str) -> str:
    """Return a light tint of a hex color (80% toward white)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = r + (255 - r) * 4 // 5
        g = g + (255 - g) * 4 // 5
        b = b + (255 - b) * 4 // 5
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:  # noqa: BLE001
        return "#E0F0FF"
