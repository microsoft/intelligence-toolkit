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
        self._run_dir: Optional[Path] = None

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
        """Build a DataFrame from the live record set, even mid-run."""
        if self._df is not None and not self._df.empty:
            return self._df
        if not self._schemify or not self._schemify.record_set:
            return None
        try:
            return self._schemify.to_dataframe()
        except Exception:  # noqa: BLE001
            return None

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
        self._run_dir = None

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
                        self.progress.entity_count = len(self._schemify.record_set.records)
                        history_count = len(
                            getattr(self._schemify, "query_history", []) or []
                        )
                        counter = int(
                            getattr(self._schemify, "_query_counter", 0) or 0
                        )
                        self.progress.query_count = max(history_count, counter, current)
                    # Snapshot partial dataset to disk so a crash doesn't lose it.
                    try:
                        self._snapshot_partial(category=category)
                    except Exception:  # noqa: BLE001
                        pass

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

    # ── On-demand verification ──────────────────────────────

    def count_unverified(self) -> tuple[int, int]:
        """Return ``(entities_with_unverified, total_unverified_values)``."""
        if not self._schemify or not self._schemify.record_set:
            return (0, 0)
        entities = 0
        values = 0
        for record in self._schemify.record_set.records:
            had = False
            for bucket in (record.attributes, record.additional_attributes):
                for av in bucket.values():
                    if not getattr(av, "verified", False):
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
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        self._snapshot_partial(
                            category=self._schemify.record_set.category
                            if self._schemify.record_set
                            else ""
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = "Verification complete"
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            pass
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
            try:
                self._snapshot_partial(
                    category=self._schemify.record_set.category
                )
            except Exception:  # noqa: BLE001
                pass
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
        except Exception:  # noqa: BLE001
            pass
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
        except Exception:  # noqa: BLE001
            pass
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
        except Exception:  # noqa: BLE001
            pass
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
        except Exception:  # noqa: BLE001
            pass
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

        def _run() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    coro = self._schemify.normalize(attributes=attributes)
                    loop.run_until_complete(coro)
                    self._df = self._schemify.to_dataframe()
                    self._dataset_json = self._build_dataset_json()
                    try:
                        self._snapshot_partial(
                            category=self._schemify.record_set.category
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    self.progress.is_running = False
                    self.progress.is_complete = True
                    self.progress.stage = "Normalization complete"
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
                self.progress.is_running = False
                self.progress.error = str(e)
                self.progress.stage = "Error"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ── Harmful content scan (I) ───────────────────────────────

    def scan_harmful_content(self) -> list[dict]:
        """Use an LLM to flag potentially harmful values across records.

        Returns a list of ``{"label", "categories", "reason", "fields"}`` dicts.
        Records with no concerns are omitted.
        """
        if not self._schemify or not self._schemify.record_set:
            return []
        try:
            from intelligence_toolkit.AI.client import OpenAIClient
            from intelligence_toolkit.AI.defaults import DEFAULT_TEMPERATURE
        except Exception:  # noqa: BLE001
            return []

        client = OpenAIClient()
        findings: list[dict] = []
        records = list(self._schemify.record_set.records)
        for record in records:
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
            if not fields:
                continue
            content = "\n".join(f"{k}: {v}" for k, v in fields.items())
            prompt = (
                "You are a content-safety classifier. Given the following "
                "structured entity record, identify any concerns across these "
                "categories: hate, harassment, violence, sexual, self-harm, "
                "illegal-activity, sensitive-PII, dangerous-instructions. "
                "If the record is benign, reply exactly: SAFE. "
                "Otherwise reply with one line of comma-separated category "
                "labels, then a newline, then a one-sentence reason.\n\n"
                f"RECORD:\n{content}\n"
            )
            try:
                resp = client.generate_chat(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=DEFAULT_TEMPERATURE,
                )
            except Exception as e:  # noqa: BLE001
                findings.append(
                    {
                        "label": record.label or "(unlabeled)",
                        "categories": ["error"],
                        "reason": f"Scan failed: {e}",
                        "fields": fields,
                    }
                )
                continue
            text = (resp or "").strip()
            if not text or text.upper().startswith("SAFE"):
                continue
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            cats_line = lines[0] if lines else ""
            reason = lines[1] if len(lines) > 1 else ""
            cats = [c.strip() for c in cats_line.split(",") if c.strip()]
            findings.append(
                {
                    "label": record.label or "(unlabeled)",
                    "categories": cats,
                    "reason": reason,
                    "fields": fields,
                }
            )
        return findings

    # ── Persistence of completed runs ──────────────────────────

    def _build_dataset_json(self) -> Optional[dict]:
        """Serialize the current record set to a JSON-able dict."""
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

    def _snapshot_partial(self, category: str) -> None:
        """Write the in-progress dataset to ``<run_dir>/data.json``."""
        if self._run_dir is None:
            return
        # Apply deterministic finalization (ALL-CAPS labels, fold
        # "Also known as" into aliases, lift units into attribute names)
        # so the live preview and on-disk snapshot match the final state.
        try:
            if self._schemify and self._schemify.record_set:
                self._schemify.resolution.finalize_normalization(
                    self._schemify.record_set
                )
        except Exception:  # noqa: BLE001
            pass
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
        except Exception:  # noqa: BLE001
            pass

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
