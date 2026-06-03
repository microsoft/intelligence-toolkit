# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build Entity Dataset API - wraps Schemify for ITK integration."""

from __future__ import annotations

import asyncio
import io
import json
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

_RUNS_DIR = Path(CACHE_PATH) / "build_entity_dataset" / "runs"


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

        The agentic discovery loop does not emit periodic
        ``on_progress`` callbacks, so the UI polls this method on each
        Streamlit rerun to surface the current query/entity counts.
        """
        if not self._schemify:
            return
        try:
            # ``_query_counter`` is only updated by the legacy iterative path;
            # the agentic loop tracks queries via ``query_history``. Use whichever
            # is larger so both code paths surface a live count.
            history_count = len(getattr(self._schemify, "query_history", []) or [])
            counter = int(getattr(self._schemify, "_query_counter", 0) or 0)
            self.progress.query_count = max(history_count, counter)
        except Exception:  # noqa: BLE001
            pass
        rs = getattr(self._schemify, "record_set", None)
        if rs is not None:
            try:
                self.progress.entity_count = len(rs.records)
            except Exception:  # noqa: BLE001
                pass
        # Live cost from the LLM client, if available.
        llm = getattr(self._schemify, "llm", None)
        if llm is not None:
            try:
                cost = float(getattr(llm, "total_cost", 0.0) or 0.0)
                tokens = int(getattr(llm, "total_tokens", 0) or 0)
                self.usage.total_cost_usd = cost
                self.usage.total_tokens = tokens
                self.usage.queries_run = self.progress.query_count
            except Exception:  # noqa: BLE001
                pass

    # ── Research lifecycle ─────────────────────────────────────

    def reset(self) -> None:
        self._schemify = None
        self.progress = ResearchProgress()
        self.usage = UsageStats()
        self._thread = None
        self._dataset_json = None
        self._df = None

    def start_research(
        self,
        api_key: str,
        category: str,
        guidance: str = "",
        schema_attributes: Optional[list[dict]] = None,
        max_queries: int = 30,
        concurrency: int = 5,
        model: str = "gpt-4o-mini",
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
                        self.progress.entity_count = len(self._schemify.record_set.records)
                        history_count = len(
                            getattr(self._schemify, "query_history", []) or []
                        )
                        counter = int(
                            getattr(self._schemify, "_query_counter", 0) or 0
                        )
                        self.progress.query_count = max(history_count, counter, current)

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
                    # Serialize to dict for JSON export
                    rs = self._schemify.record_set
                    if rs:
                        self._dataset_json = {
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
                                r.to_dict() for r in rs.records
                                if hasattr(r, "to_dict")
                            ],
                        }

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
                    if rs:
                        self.progress.entity_count = len(rs.records)

                    # Persist completed run so the UI can resume it later.
                    try:
                        self._save_run(category=category)
                    except Exception:  # noqa: BLE001
                        pass
                finally:
                    loop.close()

            except Exception as e:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                pass
        self.progress.is_running = False
        self.progress.stage = "Stopped by user"

    # ── Persistence of completed runs ──────────────────────────

    def _save_run(self, category: str) -> Optional[Path]:
        """Write the completed dataset to the local cache so it can be resumed."""
        if not self._dataset_json:
            return None
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", category.strip())[:60] or "run"
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = _RUNS_DIR / f"{ts}_{safe}"
        run_dir.mkdir(parents=True, exist_ok=True)
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

    def load_saved_run(self, data_path: str | Path) -> None:
        """Load a dataset previously written by ``_save_run``."""
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.load_dataset(data)
        self.progress = ResearchProgress(
            stage="Loaded from saved run",
            entity_count=len(data.get("records", [])),
            is_complete=True,
        )

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
    ) -> bytes:
        """Bundle dashboard.html + theme + data into a downloadable ZIP."""
        dashboard_src = (
            Path(__file__).resolve().parents[1] / "schemify" / "dashboard" / "dashboard.html"
        )

        theme = {
            "name": "custom",
            "title": title,
            "subtitle": subtitle,
            "datasetLabel": dataset_label,
            "logo": logo_filename or "",
            "logoAlt": subtitle,
            "favicon": "",
            "footer": "Built with Intelligence Toolkit.",
            "colors": {
                "primary": primary_color,
                "accent": accent_color,
                "accentSoft": _lighten(accent_color),
                "accent2": accent_color,
                "warn": "#E85D4A",
                "highlight": "#F0A830",
                "success": "#4CAF50",
                "bg": "#F5F7FA",
                "surface": "#FFFFFF",
                "text": "#1A2A3A",
                "muted": "#6B7C8D",
                "border": "#DDE3EA",
            },
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
