# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Verifying proposed review corrections against fresh public sources."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
import time
from pathlib import Path
from uuid import uuid4

from . import state, storage
from .state import ResearchProgress, Session

_logger = logging.getLogger(__name__)


def review_verification_report(session: Session) -> dict:
    with session.state_lock:
        return copy.deepcopy(session.review_verification_report)


def review_dataset_fingerprint(session: Session) -> str:
    from itk_schemify.review_verification import fingerprint

    with session.state_lock:
        return fingerprint(storage.build_dataset_json(session) or session.dataset_json)


def list_review_verifications(session: Session) -> list[dict]:
    """List persisted captures for the loaded dataset, without changing it."""
    from itk_schemify.review_verification import read_report

    root = (session.run_dir or session.runs_dir) / "review_verification"
    current = review_dataset_fingerprint(session)
    runs = []
    for path in sorted(root.glob("*/report.json"), reverse=True):
        try:
            report = read_report(path.parent)
            if report["dataset_fingerprint"] == current:
                runs.append({
                    "directory": str(path.parent),
                    "started_at": report["started_at"],
                    "backend": report["backend"],
                    "complete": report["complete"],
                    "summary": report["summary"],
                })
        except (OSError, ValueError, KeyError) as exc:
            _logger.warning("Could not read review verification %s: %s", path, exc)
    return runs


def open_review_verification(session: Session, directory: str) -> None:
    from itk_schemify.review_verification import read_report

    if state.is_running(session):
        msg = "Wait for the active job before opening a verification report"
        raise ValueError(msg)
    path = Path(directory)
    report = read_report(path)
    with session.state_lock:
        session.review_verification_report = report
        session.review_verification_dir = path


def review_verification_bundle(session: Session) -> bytes:
    from itk_schemify.review_verification import evidence_bundle

    with session.state_lock:
        if state.is_verifying_review(session) or not session.review_verification_dir:
            msg = "Open a completed or paused verification before exporting evidence"
            raise ValueError(msg)
        return evidence_bundle(session.review_verification_dir)


def stop_review_verification(session: Session) -> None:
    if not state.is_verifying_review(session):
        msg = "No review verification is running"
        raise ValueError(msg)
    session.review_verification_stop = True


def start_review_verification(
    session: Session,
    proposals: list[dict],
    *,
    backend: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    concurrency: int = 4,
    max_sources: int = 12,
    resume_directory: str | None = None,
    retry_failed_sources: bool = False,
) -> None:
    """Verify an immutable snapshot, including loaded read-only datasets."""
    from itk_schemify.review_verification import (
        bind_proposals,
        make_client,
        verify_proposals,
    )

    with session.state_lock:
        if state.is_running(session):
            msg = "Another dataset job is already running"
            raise ValueError(msg)
        snapshot = copy.deepcopy(
            storage.build_dataset_json(session) or session.dataset_json
        )
        if snapshot is None:
            msg = "Load a dataset before verifying review decisions"
            raise ValueError(msg)
        bound = bind_proposals(snapshot, proposals)
        client = make_client(backend, api_key=api_key, model=model)
        directory = (
            Path(resume_directory)
            if resume_directory
            else (
                (session.run_dir or session.runs_dir)
                / "review_verification"
                / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            )
        )
        session.review_verification_dir = directory
        session.review_verification_report = {}
        session.review_verification_stop = False
        session.review_verification_active = True
        session.review_verification_progress = ResearchProgress(
            stage="Verifying review decisions",
            is_running=True,
            total=len(bound),
        )

        def progress(report: dict) -> None:
            with session.state_lock:
                session.review_verification_report = report
                session.review_verification_progress.current = report["summary"][
                    "completed_records"
                ]

        def run() -> None:
            from jsonschema import ValidationError
            from openai import OpenAIError

            try:
                report = asyncio.run(
                    verify_proposals(
                        snapshot,
                        bound,
                        directory,
                        client,
                        backend=backend,
                        concurrency=concurrency,
                        max_sources=max_sources,
                        resume=resume_directory is not None,
                        on_progress=progress,
                        retry_failed_sources=retry_failed_sources,
                        should_stop=lambda: session.review_verification_stop,
                    )
                )
                with session.state_lock:
                    session.review_verification_report = report
                    session.review_verification_progress.is_complete = report[
                        "complete"
                    ]
                    session.review_verification_progress.stage = (
                        "Review verification complete; no changes applied"
                        if report["complete"]
                        else "Review verification paused; checkpoints saved"
                    )
                    errors = report["summary"]["processing_errors"]
                    if errors:
                        session.review_verification_progress.error = f"{errors} records have processing/retrieval errors; inspect the report"
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                ValidationError,
                OpenAIError,
            ) as exc:
                _logger.exception("Review verification failed; checkpoints retained")
                session.review_verification_progress.error = str(exc)
                session.review_verification_progress.stage = (
                    "Review verification failed"
                )
            finally:
                with session.state_lock:
                    session.review_verification_active = False
                    session.review_verification_progress.is_running = False
                    if (
                        session.review_verification_progress.stage
                        == "Verifying review decisions"
                    ):
                        _logger.error(
                            "Verification terminated unexpectedly", exc_info=True
                        )
                        session.review_verification_progress.error = (
                            "Verification terminated unexpectedly; inspect application logs. "
                            "Existing checkpoints are retained."
                        )

        session.thread = threading.Thread(target=run, daemon=True)
        session.thread.start()


def resume_review_verification(
    session: Session,
    directory: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    concurrency: int = 4,
    retry_failed_sources: bool = False,
) -> None:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    proposals = json.loads((path / "proposals.json").read_text(encoding="utf-8"))[
        "items"
    ]
    start_review_verification(
        session,
        proposals,
        backend=manifest["backend"],
        api_key=api_key,
        model=model if model is not None else manifest.get("requested_model"),
        concurrency=concurrency,
        max_sources=manifest["max_sources"],
        resume_directory=directory,
        retry_failed_sources=retry_failed_sources,
    )
