# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Serialising the record set, snapshots and saved runs, and loading them back."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd

from . import config, state
from .state import ResearchProgress, Session, UsageStats

_logger = logging.getLogger(__name__)


# Minimum seconds between disk snapshots during a running extraction.
# Progress callbacks fire on every query/extraction tick; without
# throttling that's a blocking JSON write per tick.
_SNAPSHOT_MIN_INTERVAL_SECONDS = 10.0


def post_curation_refresh(session: Session) -> None:
    """Refresh derived state after any merge/split/rename edit."""
    if not session.schemify or not session.schemify.record_set:
        return
    rs = session.schemify.record_set
    try:
        session.schemify.resolution.finalize_normalization(rs)
    except Exception as e:  # noqa: BLE001
        _logger.warning("finalize_normalization after curation failed: %s", e)
    try:
        rs.update_schema_frequencies()
    except Exception as e:  # noqa: BLE001
        _logger.warning("update_schema_frequencies failed: %s", e)
    try:
        session.df = session.schemify.to_dataframe()
    except Exception as e:  # noqa: BLE001
        _logger.warning("to_dataframe after curation failed: %s", e)
    session.dataset_json = build_dataset_json(session)
    state.bump_record_version(session)
    try:
        snapshot_partial(session, category=rs.category)
    except Exception as e:  # noqa: BLE001
        _logger.warning("snapshot after curation failed: %s", e)


def import_dataset(
    session: Session,
    data: dict,
    *,
    api_key: str | None = None,
    model: str = config.DEFAULT_MODEL,
    budget: float = 10.0,
) -> Path:
    """Import a dataset JSON as a new saved run so it can be edited, extended and re-published.

    Without ``api_key`` the dataset loads read-only (same as :meth:`load_saved_run`).
    """
    session.runs_dir.mkdir(parents=True, exist_ok=True)
    category = (data.get("category") or "imported").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category)[:60] or "run"
    run_dir = session.runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_import_{safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "data.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "category": category,
                "timestamp": run_dir.name.split("_", 1)[0],
                "entity_count": len(data.get("records", [])),
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "queries_run": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    load_saved_run(session, path, api_key=api_key, model=model, budget=budget)
    return path


def build_dataset_json(session: Session) -> dict | None:
    """Serialize the current record set to a JSON-able dict.

    ``finalize_normalization`` (called from ``_snapshot_partial`` and
    ``Schemify.finalize``) is responsible for pruning empty schema
    attributes and placeholder values. This builder just serializes
    the current state — no extra filtering here, so the dashboard,
    CSV, and on-screen dataframe all see the same columns.
    """
    rs = getattr(session.schemify, "record_set", None) if session.schemify else None
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
                "required": getattr(a, "required", False),
                "frequency": getattr(a, "frequency", 0.0),
                "provisional_values": getattr(a, "provisional_values", []),
                "canonical_values": getattr(a, "canonical_values", []),
                "canonical_value_descriptions": getattr(
                    a, "canonical_value_descriptions", {}
                )
                or {},
                "locked": bool(getattr(a, "locked", False)),
            }
            for a in rs.schema_attributes
        ],
        "records": [r.to_dict() for r in rs.records if hasattr(r, "to_dict")],
        "user_exclusions": list(getattr(rs, "user_exclusions", []) or []),
        "do_not_merge": [
            sorted(list(p)) for p in (getattr(rs, "do_not_merge", None) or set())
        ],
        "history": list(getattr(rs, "history", []) or []),
        "created_at": rs.created_at.isoformat()
        if getattr(rs, "created_at", None)
        else None,
    }


def snapshot_partial(
    session: Session, category: str, *, throttle: bool = False
) -> None:
    """Write the in-progress dataset to ``<run_dir>/data.json``.

    Pass ``throttle=True`` from the per-query progress callback to
    rate-limit writes to ``_SNAPSHOT_MIN_INTERVAL_SECONDS``. User-
    initiated mutations (curation, exclusions, etc.) use the default
    and persist immediately.
    """
    if session.run_dir is None:
        return
    if throttle:
        now = time.monotonic()
        if (now - session.last_snapshot_time) < _SNAPSHOT_MIN_INTERVAL_SECONDS:
            session.snapshot_pending = True
            return
        session.last_snapshot_time = now
        session.snapshot_pending = False
    else:
        session.last_snapshot_time = time.monotonic()
        session.snapshot_pending = False
    # Apply deterministic finalization (ALL-CAPS labels, fold
    # "Also known as" into aliases, lift units into attribute names)
    # so the live preview and on-disk snapshot match the final state.
    try:
        if session.schemify and session.schemify.record_set:
            session.schemify.resolution.finalize_normalization(
                session.schemify.record_set
            )
    except Exception as e:  # noqa: BLE001
        _logger.warning("finalize_normalization failed during snapshot: %s", e)
    data = build_dataset_json(session)
    if not data:
        return
    try:
        (session.run_dir / "data.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ts = session.run_dir.name.split("_", 1)[0]
        meta = {
            "category": category,
            "timestamp": ts,
            "entity_count": len(data.get("records", [])),
            **_cumulative_usage(session),
            "in_progress": True,
        }
        (session.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    except OSError as e:
        _logger.warning("snapshot write failed: %s", e)


def save_run(session: Session, category: str) -> Path | None:
    """Write the completed dataset to the run directory so it can be resumed."""
    if not session.dataset_json:
        return None
    if session.run_dir is None:
        session.runs_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
        ts = time.strftime("%Y%m%d-%H%M%S")
        session.run_dir = session.runs_dir / f"{ts}_{safe}"
        session.run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = session.run_dir.name.split("_", 1)[0]
    run_dir = session.run_dir
    data_path = run_dir / "data.json"
    data_path.write_text(
        json.dumps(session.dataset_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "category": category,
        "timestamp": ts,
        "entity_count": len(session.dataset_json.get("records", [])),
        **_cumulative_usage(session),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return data_path


def _cumulative_usage(session: Session) -> dict:
    b, u = session.usage_baseline, session.usage
    return {
        "total_tokens": b.total_tokens + u.total_tokens,
        "total_cost_usd": round(b.total_cost_usd + u.total_cost_usd, 4),
        "queries_run": b.queries_run + u.queries_run,
    }


def list_saved_runs(runs_dir: Path | None = None) -> list[dict]:
    """Return saved runs under ``runs_dir`` (default :data:`itk_entities.state.RUNS_DIR`), newest first."""
    root = runs_dir if runs_dir is not None else state.RUNS_DIR
    if not root.exists():
        return []
    runs: list[dict] = []
    for run_dir in root.iterdir():
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
        runs.append({
            "name": run_dir.name,
            "path": str(data_path),
            "category": meta.get("category", ""),
            "timestamp": meta.get("timestamp", run_dir.name),
            "entity_count": meta.get("entity_count", 0),
            "total_cost_usd": meta.get("total_cost_usd", 0.0),
            "queries_run": meta.get("queries_run", 0),
        })
    runs.sort(key=lambda r: r["name"], reverse=True)
    return runs


def load_saved_run(
    session: Session,
    data_path: str | Path,
    api_key: str | None = None,
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
    load_dataset(session, data)

    # Point the run dir at the loaded run so further snapshots / saves
    # accumulate alongside the original data.json + meta.json.
    try:
        session.run_dir = Path(data_path).parent
    except (TypeError, ValueError) as e:
        _logger.warning("could not resolve run_dir from %s: %s", data_path, e)
        session.run_dir = None

    session.usage = UsageStats()
    session.usage_baseline = UsageStats()
    meta_path = session.run_dir / "meta.json" if session.run_dir else None
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            session.usage_baseline = UsageStats(
                total_tokens=int(meta.get("total_tokens", 0) or 0),
                total_cost_usd=float(meta.get("total_cost_usd", 0.0) or 0.0),
                queries_run=int(meta.get("queries_run", 0) or 0),
            )
        except (OSError, ValueError) as e:
            _logger.warning("could not read usage from %s: %s", meta_path, e)

    session.read_only_reason = None
    if api_key:
        try:
            _rehydrate_schemify(
                session, data, api_key=api_key, model=model, budget=budget
            )
        except Exception as e:
            _logger.exception("rehydrate_schemify failed")
            session.schemify = None
            session.read_only_reason = (
                f"Failed to rehydrate live research state ({e}). "
                "Reload with a valid API key to continue research."
            )
    else:
        session.schemify = None
        session.read_only_reason = (
            "Loaded without an API key — dataset is read-only. "
            "Provide an API key in Settings and reload this run to "
            "continue research or curate aliases."
        )
    state.bump_record_version(session)

    entity_count = len(data.get("records", []))
    session.progress = ResearchProgress(
        stage="Loaded from saved run",
        entity_count=entity_count,
        query_count=int((session.dataset_json or {}).get("query_count", 0) or 0),
        is_complete=True,
    )


def _rehydrate_schemify(
    session: Session,
    data: dict,
    *,
    api_key: str,
    model: str = config.DEFAULT_MODEL,
    budget: float = 10.0,
) -> None:
    """Build a fresh Schemify and seed it with ``data`` so the loaded
    run can be continued, seeded and curated.
    """
    from itk_schemify import Schemify
    from itk_schemify.models import (
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
    session.schemify = sch


def load_dataset(session: Session, data: dict) -> None:
    """Load a previously saved dataset JSON into this object."""
    from itk_schemify.models import _clean_title

    if state.is_verifying_review(session):
        msg = "Stop review verification before loading a different dataset"
        raise ValueError(msg)
    session.review_verification_report = {}
    session.review_verification_dir = None

    for r in data.get("records", []):
        for bucket in ("attributes", "additional_attributes"):
            for av in (r.get(bucket) or {}).values():
                for sv in (av.get("values") or []) if isinstance(av, dict) else []:
                    for s in sv.get("sources") or []:
                        if isinstance(s, dict) and s.get("title"):
                            s["title"] = _clean_title(s["title"])
    session.dataset_json = data
    records = data.get("records", [])
    attrs = [a.get("name", "") for a in data.get("schema_attributes", [])]
    rows = []
    for r in records:
        row: dict = {"name": r.get("label", r.get("name", ""))}
        for attr in attrs:
            raw = r.get("attributes", {}).get(attr, {})
            vals = raw.get("values", []) if isinstance(raw, dict) else []
            row[attr] = "; ".join(
                v.get("value", "") for v in vals if isinstance(v, dict)
            )
        rows.append(row)
    session.df = pd.DataFrame(rows) if rows else pd.DataFrame()


def get_dataset_bytes_json(session: Session) -> bytes:
    if session.dataset_json:
        return json.dumps(session.dataset_json, indent=2, ensure_ascii=False).encode()
    return b"{}"


def get_dataset_bytes_csv(session: Session) -> bytes:
    if session.df is not None and not session.df.empty:
        return session.df.to_csv(index=False).encode()
    return b""
