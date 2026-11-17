# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Parity of the model-assisted operations with the legacy class.

Audits, alias suggestions, single-value selection, recategorization and the
relevance filter all ask a model. The model is replaced by the same canned
replies on both sides, so what is compared is everything around the call:
which records are sent, how replies are applied, and what is reported back.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import itk_core.client
import itk_schemify.extraction as new_extraction
import itk_schemify.llm as new_llm
import pytest
from itk_entities import BuildEntityDataset
from test_entities_legacy_parity import _normalize, sample_dataset

legacy_api = pytest.importorskip("intelligence_toolkit.build_entity_dataset.api")
legacy_extraction = pytest.importorskip("intelligence_toolkit.schemify.extraction")
legacy_llm = pytest.importorskip("intelligence_toolkit.schemify.llm")

ALIAS_REPLY = json.dumps([
    {
        "primary": "ACME TRACKER",
        "members": ["ACME TRACKER PRO", "NOT A LABEL"],
        "reason": "same product",
    }
])

STRUCTURED_REPLY = {
    "results": [
        {
            "label": "ACME TRACKER",
            "in_scope": True,
            "confidence": 0.9,
            "reason": "a tool",
        },
        {
            "label": "DELTA",
            "in_scope": False,
            "confidence": 0.95,
            "reason": "not a tool",
        },
    ],
    "duplicates": [],
    "groups": [],
    "proposals": [],
    "choices": [],
    "attributes": [],
    "remappings": [],
    "values": [],
    "languages": [{"code": "es", "name": "Spanish", "reason": "regional coverage"}],
    "reasoning": "canned",
}


class FakeChatClient:
    """Stands in for itk-core's chat client, which both implementations import."""

    def __init__(self, *_, **__):
        pass

    def generate_chat(self, messages, *_, **__):
        text = json.dumps(messages, default=str)
        return (
            ALIAS_REPLY
            if "DISTINCT records" in text
            else '{"flagged": false, "categories": []}'
        )


def _stub_models(monkeypatch) -> None:
    async def structured_completion(self, *_, **__):
        return json.loads(json.dumps(STRUCTURED_REPLY))

    async def expand_record(self, record, record_set, **_):
        return record

    monkeypatch.setattr(itk_core.client, "OpenAIClient", FakeChatClient)
    for llm_module, extraction_module in (
        (legacy_llm, legacy_extraction),
        (new_llm, new_extraction),
    ):
        monkeypatch.setattr(
            llm_module.LLMClient, "structured_completion", structured_completion
        )
        monkeypatch.setattr(
            extraction_module.ExtractionEngine, "expand_record", expand_record
        )


def _wait(api: Any) -> None:
    deadline = time.monotonic() + 60
    while api.is_running:
        assert time.monotonic() < deadline, "worker did not finish"
        time.sleep(0.02)


def _threaded(
    start: Callable[[Any], Any], report: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    def step(api: Any) -> Any:
        start(api)
        _wait(api)
        return report(api)

    return step


RELEVANCE = {
    "criterion": "is a case management tool",
    "evidence_attribute": "Type",
    "connections": ["direct", "indirect"],
}

SCRIPT: list[tuple[str, Callable[[Any], Any]]] = [
    ("suggest_alias_groups", lambda a: a.suggest_alias_groups()),
    ("scan_harmful_content", lambda a: a.scan_harmful_content(concurrency=1)),
    ("audit_scope", lambda a: a.audit_scope(concurrency=1)),
    ("audit_relevance", lambda a: a.audit_relevance(concurrency=1, **RELEVANCE)),
    ("audit_merge_quality", lambda a: a.audit_merge_quality(concurrency=1)),
    ("audit_duplicate_clusters", lambda a: a.audit_duplicate_clusters(concurrency=1)),
    (
        "select_single_values",
        lambda a: a.select_single_values("Country", concurrency=1),
    ),
    (
        "propose_recategorization",
        lambda a: a.propose_recategorization("Split Type in two", attributes=["Type"]),
    ),
    (
        "apply_recategorization",
        lambda a: a.apply_recategorization({
            "attributes": [
                {
                    "name": "Type",
                    "description": "Kind of tool",
                    "canonical_values": ["Hotline", "Records System"],
                    "canonical_value_descriptions": {
                        "Records System": "stores case records"
                    },
                    "remappings": [{"from": "Database", "to": ["Records System"]}],
                }
            ],
            "out_of_scope": [],
        }),
    ),
    (
        "start_recategorization_search",
        _threaded(
            lambda a: a.start_recategorization_search(["Type"], concurrency=1),
            lambda a: (a.progress, a.recat_added),
        ),
    ),
    (
        "start_relevance_filter",
        _threaded(
            lambda a: a.start_relevance_filter(research_evidence=True, **RELEVANCE),
            lambda a: (a.progress, a.relevance_results),
        ),
    ),
    (
        "apply_dedup_merge",
        lambda a: a.apply_dedup_merge(
            "ACME TRACKER", ["ACME TRACKER PRO", "NOPE"], "same product"
        ),
    ),
    (
        "apply_audit_split",
        lambda a: a.apply_audit_split(
            "ACME TRACKER",
            {
                "verdict": "split",
                "proposed_splits": [{"label": "ACME TRACKER LITE", "aliases": ["AT"]}],
            },
        ),
    ),
    (
        "propose_search_languages",
        lambda a: a.propose_search_languages(api_key="test", category="Tools"),
    ),
]


def _run_script(api: Any, root: Any) -> list[tuple[str, Any, Any]]:
    trace = []
    for name, step in SCRIPT:
        try:
            outcome: Any = ("ok", _normalize(step(api), root))
        except Exception as error:  # noqa: BLE001 - an error is a result to compare too
            outcome = ("error", type(error).__name__, _normalize(str(error), root))
        trace.append((name, outcome, _normalize(api.dataset_json, root)))
    return trace


@pytest.fixture()
def traces(tmp_path, monkeypatch):
    monkeypatch.chdir(
        tmp_path
    )  # Schemify creates its cache folder in the working directory
    _stub_models(monkeypatch)
    legacy_root = tmp_path / "legacy"
    new_root = tmp_path / "new"
    monkeypatch.setattr(legacy_api, "_RUNS_DIR", legacy_root)

    legacy = legacy_api.BuildEntityDataset()
    legacy.import_dataset(sample_dataset(), api_key="test")
    new = BuildEntityDataset(runs_dir=new_root)
    new.import_dataset(sample_dataset(), api_key="test")

    return _run_script(legacy, legacy_root), _run_script(new, new_root)


def test_model_assisted_steps_match_the_legacy_class(traces):
    legacy_trace, new_trace = traces

    for (name, legacy_outcome, legacy_data), (_, new_outcome, new_data) in zip(
        legacy_trace, new_trace, strict=True
    ):
        assert new_outcome == legacy_outcome, f"result of step '{name}' differs"
        assert new_data == legacy_data, f"dataset after step '{name}' differs"


def test_most_model_assisted_steps_complete(traces):
    """Guard against a script that passes only because every step fails the same way."""
    _, new_trace = traces

    errors = [
        f"{name}: {outcome[1:]}"
        for name, outcome, _ in new_trace
        if outcome[0] == "error"
    ]
    assert len(errors) <= 3, errors
