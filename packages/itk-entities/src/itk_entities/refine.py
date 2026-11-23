# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Passes over an existing dataset: verify unsourced values, complete gaps, normalize."""

from __future__ import annotations

import asyncio
import logging
import threading

from . import state, storage
from .state import ResearchProgress, Session, UsageStats

_logger = logging.getLogger(__name__)


def start_verification(session: Session, concurrency: int = 12) -> None:
    """Run verification in a background daemon thread."""
    if state.is_running(session) or not session.schemify:
        return
    total_entities = (
        len(session.schemify.record_set.records) if session.schemify.record_set else 0
    )
    session.progress = ResearchProgress(
        is_running=True,
        stage="Verifying attribute values…",
        entity_count=total_entities,
    )

    def _on_verify_progress(done: int, total: int, label: str) -> None:
        # Called from the verification event loop thread. Mutating the
        # dataclass fields is fine — the UI polls them.
        session.progress.current = done
        session.progress.total = total
        session.progress.stage = (
            f"Verifying {done}/{total}: {label}"
            if label
            else f"Verifying {done}/{total}…"
        )

    def _run() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    session.schemify.verify_unverified(
                        concurrency=concurrency,
                        progress_callback=_on_verify_progress,
                    )
                )
                session.df = session.schemify.to_dataframe()
                session.dataset_json = storage.build_dataset_json(session)
                state.bump_record_version(session)
                # Refresh usage figures.
                try:
                    stats = session.schemify.get_stats()
                    llm_usage = stats.get("llm_usage", {}) or {}
                    session.usage = UsageStats(
                        total_tokens=int(llm_usage.get("total_tokens", 0) or 0),
                        total_cost_usd=float(
                            llm_usage.get("total_cost_usd", 0.0) or 0.0
                        ),
                        queries_run=session.progress.query_count,
                    )
                except Exception as e:  # noqa: BLE001
                    _logger.warning("usage refresh after verify failed: %s", e)
                try:
                    storage.snapshot_partial(
                        session,
                        category=session.schemify.record_set.category
                        if session.schemify.record_set
                        else "",
                    )
                except Exception as e:  # noqa: BLE001
                    _logger.warning("snapshot after verify failed: %s", e)
                session.progress.is_running = False
                session.progress.is_complete = True
                session.progress.stage = "Verification complete"
            finally:
                loop.close()
        except Exception as e:
            _logger.exception("verification worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()


def start_normalize(session: Session, attributes: list[str] | None = None) -> None:
    """Run schemify.normalize in a background thread for the given attributes."""
    if state.is_running(session) or not session.schemify:
        return
    target = ", ".join(attributes) if attributes else "all attributes"
    session.progress = ResearchProgress(
        is_running=True,
        stage=f"Normalizing {target}…",
        entity_count=len(session.schemify.record_set.records)
        if session.schemify.record_set
        else 0,
    )

    def _on_norm_progress(done: int, total: int, attr_name: str) -> None:
        session.progress.current = done
        session.progress.total = total
        if attr_name:
            session.progress.stage = f"Normalizing {attr_name} ({done}/{total})"
        else:
            session.progress.stage = f"Normalizing ({done}/{total})"

    def _run() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Pass progress_callback when supported; otherwise fall
                # back so older Schemify still works.
                try:
                    coro = session.schemify.normalize(
                        attributes=attributes,
                        progress_callback=_on_norm_progress,
                    )
                except TypeError:
                    coro = session.schemify.normalize(attributes=attributes)
                loop.run_until_complete(coro)
                session.df = session.schemify.to_dataframe()
                session.dataset_json = storage.build_dataset_json(session)
                state.bump_record_version(session)
                try:
                    storage.snapshot_partial(
                        session, category=session.schemify.record_set.category
                    )
                except Exception as e:  # noqa: BLE001
                    _logger.warning("snapshot after normalize failed: %s", e)
                session.progress.is_running = False
                session.progress.is_complete = True
                session.progress.stage = "Normalization complete"
            finally:
                loop.close()
        except Exception as e:
            _logger.exception("normalize worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()


def start_completion(session: Session, concurrency: int = 8) -> None:
    """Fill missing schema attributes for every record (background)."""
    if (
        state.is_running(session)
        or not session.schemify
        or not session.schemify.record_set
    ):
        return
    if getattr(session.schemify, "llm", None) is None:
        raise ValueError("Completion requires a live session with an API key.")
    session.progress = ResearchProgress(
        is_running=True,
        stage="Filling missing attributes…",
        entity_count=len(session.schemify.record_set.records),
    )

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                session.schemify.complete_all(concurrency=concurrency, verbose=False)
            )
            storage.post_curation_refresh(session)
            session.progress.is_running = False
            session.progress.is_complete = True
            session.progress.stage = "Completion pass complete"
        except Exception as e:
            _logger.exception("completion worker failed")
            session.progress.is_running = False
            session.progress.error = str(e)
            session.progress.stage = "Error"
        finally:
            loop.close()

    session.thread = threading.Thread(target=_run, daemon=True)
    session.thread.start()
