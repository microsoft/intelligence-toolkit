# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Parity of the background-run workers with the legacy class.

Research, verification, completion, normalization, auto mode and the
rediscovery benchmark all run Schemify in a worker thread. Here Schemify's
network-bound steps are replaced by the same deterministic stand-ins on both
sides, so what is compared is the orchestration around them: progress,
usage, the dataset, and what is written to the run folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import itk_schemify.llm as new_llm
import itk_schemify.models as new_models
import itk_schemify.schemify as new_schemify
import pytest
from itk_entities import BuildEntityDataset
from test_entities_legacy_parity import _normalize

legacy_api = pytest.importorskip("intelligence_toolkit.build_entity_dataset.api")
legacy_llm = pytest.importorskip("intelligence_toolkit.schemify.llm")
legacy_models = pytest.importorskip("intelligence_toolkit.schemify.models")
legacy_schemify = pytest.importorskip("intelligence_toolkit.schemify.schemify")

JUDGE_REPLY = {
    "complete": True,
    "is_complete": True,
    "confidence": 0.9,
    "reason": "coverage looks sufficient",
    "missing_gaps": [],
    "languages": [],
}


def _stub_schemify(monkeypatch, schemify_module, models, llm_module) -> None:
    """Replace the steps that would search the web or call a model."""

    def sourced(value: str, url: str) -> Any:
        return models.AttributeValue(
            values=[
                models.SourcedValue(
                    value=value, sources=[models.Citation(url=url, title="t")]
                )
            ]
        )

    async def initialize(self, category, guidance="", schema_attributes=None, **_):
        self.record_set = models.RecordSet(
            category=category,
            guidance=guidance,
            schema_attributes=list(
                schema_attributes or [models.SchemaAttribute(name="Country")]
            ),
        )
        self._initialized = True
        return self

    async def run_agentic(self, max_queries=100, seed_records=None, **_):
        records = self.record_set.records
        known = {r.label for r in records}
        for seed in seed_records or []:
            if seed["label"] not in known:
                records.append(models.Record(label=seed["label"]))
                known.add(seed["label"])
        number = len(records) + 1
        records.append(
            models.Record(
                label=f"FOUND {number}",
                attributes={
                    "Country": sourced("USA", f"https://found.example/{number}")
                },
            )
        )
        self._query_counter += max_queries
        if self._on_progress:
            self._on_progress("Discovery", max_queries, max_queries)
        return self.to_dataframe()

    async def complete_all(self, **_):
        filled = 0
        for record in self.record_set.records:
            if "Country" not in record.attributes:
                record.attributes["Country"] = models.AttributeValue(
                    values=[models.SourcedValue(value="Unknown", sources=[])]
                )
                filled += 1
        return {
            "entities_processed": filled,
            "total_calls": filled,
            "attributes_filled": filled,
            "still_missing_count": 0,
        }

    async def verify_unverified(self, progress_callback=None, **_):
        verified = 0
        for record in self.record_set.records:
            attribute = record.attributes.get("Country")
            if attribute and attribute.values and not attribute.values[0].sources:
                attribute.values[0].sources.append(
                    models.Citation(
                        url=f"https://verify.example/{record.label}", title="t"
                    )
                )
                verified += 1
                if progress_callback:
                    progress_callback(verified, verified, record.label)
        return {"entities_checked": verified, "values_verified": verified, "errors": 0}

    async def normalize(self, attributes=None, progress_callback=None, **_):
        if progress_callback:
            progress_callback(1, 1, "Country")
        return {"Country": {"n_changed": 0}}

    async def structured_completion(self, *_, **__):
        return dict(JUDGE_REPLY)

    for name, stub in {
        "initialize": initialize,
        "run_agentic": run_agentic,
        "complete_all": complete_all,
        "verify_unverified": verify_unverified,
        "normalize": normalize,
    }.items():
        monkeypatch.setattr(schemify_module.Schemify, name, stub)
    monkeypatch.setattr(
        llm_module.LLMClient, "structured_completion", structured_completion
    )


def _wait(api: Any) -> None:
    deadline = time.monotonic() + 60
    while api.is_running:
        assert time.monotonic() < deadline, "worker did not finish"
        time.sleep(0.02)


def _run_folder(api: Any) -> dict:
    folder = api.run_dir
    if folder is None:
        return {}
    listing = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            relative = path.relative_to(folder).as_posix()
            listing[relative] = (
                json.loads(path.read_text(encoding="utf-8"))
                if relative in ("data.json", "meta.json")
                else "<file>"
            )
    return listing


def _capture(api: Any, root: Path, result: Any = None) -> dict:
    return _normalize(
        {
            "result": result,
            "progress": api.progress,
            "usage": api.usage,
            "dataset": api.dataset_json,
            "dataframe": api.current_dataframe(),
            "unverified": api.count_unverified(),
            "run_folder": _run_folder(api),
        },
        root,
    )


def _manual_run(api: Any, root: Path) -> list[tuple[str, dict]]:
    trace = []

    def step(name: str, result: Any = None) -> None:
        _wait(api)
        trace.append((name, _capture(api, root, result)))

    api.start_research(
        api_key="test",
        category="Tools",
        guidance="NGO tools",
        schema_attributes=[{"name": "Country", "description": "Where it is based"}],
        max_queries=3,
        seed_labels=["Seed A", "Seed A", "Seed B"],
        complete=True,
        verify=True,
    )
    step("start_research")
    step("continue_research", api.continue_research(max_queries=2, verify=True))
    api.add_candidate_entities(["Seed C"])
    step("add_candidate_entities")
    api.start_completion(concurrency=2)
    step("start_completion")
    api.start_verification(concurrency=2)
    step("start_verification")
    api.start_normalize(attributes=["Country"])
    step("start_normalize")
    api.stop_research()
    step("stop_research")
    return trace


def _auto_run(api: Any, root: Path) -> list[tuple[str, dict]]:
    api.start_auto_mode(
        api_key="test",
        category="Tools",
        guidance="NGO tools",
        schema_attributes=[{"name": "Country", "description": "Where it is based"}],
        reference_labels=["Seed A", "Seed B"],
        max_iterations=2,
        per_iter_query_budget=2,
        min_iterations=1,
    )
    _wait(api)
    return [("start_auto_mode", _capture(api, root))]


def _benchmark_run(api: Any, root: Path) -> list[tuple[str, dict]]:
    api.start_rediscovery_benchmark(
        api_key="test",
        category="Tools",
        guidance="NGO tools",
        schema_attributes=[{"name": "Country", "description": "Where it is based"}],
        reference_labels=["FOUND 1", "Never Found"],
        max_iterations=1,
        per_iter_query_budget=2,
        gap_fill_query_budget=2,
        min_iterations=1,
    )
    _wait(api)
    return [("start_rediscovery_benchmark", _capture(api, root))]


@pytest.fixture()
def both(tmp_path, monkeypatch):
    monkeypatch.chdir(
        tmp_path
    )  # Schemify creates its cache folder in the working directory
    _stub_schemify(monkeypatch, legacy_schemify, legacy_models, legacy_llm)
    _stub_schemify(monkeypatch, new_schemify, new_models, new_llm)
    legacy_root = tmp_path / "legacy"
    new_root = tmp_path / "new"
    monkeypatch.setattr(legacy_api, "_RUNS_DIR", legacy_root)
    return (legacy_api.BuildEntityDataset(), legacy_root), (
        BuildEntityDataset(runs_dir=new_root),
        new_root,
    )


@pytest.mark.parametrize("scenario", [_manual_run, _auto_run, _benchmark_run])
def test_run_workers_match_the_legacy_class(both, scenario):
    (legacy, legacy_root), (new, new_root) = both

    legacy_trace = scenario(legacy, legacy_root)
    new_trace = scenario(new, new_root)

    for (name, legacy_state), (_, new_state) in zip(
        legacy_trace, new_trace, strict=True
    ):
        for key in legacy_state:
            assert new_state[key] == legacy_state[key], (
                f"'{key}' after '{name}' differs"
            )
        assert not new_state["progress"]["error"], (
            f"'{name}' failed: {new_state['progress']['error']}"
        )
        assert new_state["dataset"]["records"], f"'{name}' produced no records"
