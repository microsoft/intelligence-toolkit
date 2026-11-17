# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Guards on a run: the budget cap, duplicate detection, prompt text and completion retries."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from itk_schemify import Schemify, scope_audit
from itk_schemify.llm import LLMClient
from itk_schemify.models import (
    BudgetExceededError,
    Record,
    RecordSet,
    SchemaAttribute,
    SchemifyConfig,
)
from itk_schemify.resolution import ResolutionEngine, find_fuzzy_match


def _config(**overrides) -> SchemifyConfig:
    settings = {
        "api_key": "test",
        "search_model": "gpt-5.4-mini",
        "completion_model": "gpt-5.4-mini",
        "cache_enabled": False,
    }
    settings.update(overrides)
    return SchemifyConfig(**settings)


class FakeResponses:
    """Stands in for itk-core's client: records each request, replies with fixed JSON."""

    def __init__(self, reply: dict, input_tokens: int = 1_000_000):
        self.requests: list[dict] = []
        self._reply = reply
        self._input_tokens = input_tokens

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        text = SimpleNamespace(type="output_text", text=json.dumps(self._reply), annotations=[])
        return SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[text])],
            usage=SimpleNamespace(
                input_tokens=self._input_tokens,
                output_tokens=0,
                total_tokens=self._input_tokens,
            ),
        )


def _llm(reply: dict | None = None, **config) -> tuple[LLMClient, FakeResponses]:
    llm = LLMClient(_config(**config))
    responses = FakeResponses(reply or {"ok": True})
    llm.client = SimpleNamespace(generate_response_async=responses.create)
    return llm, responses


# --- budget cap -------------------------------------------------------------


def test_response_that_crosses_the_budget_is_returned_then_no_more_requests_are_sent():
    # 1M input tokens at $0.75/M per call against a $0.50 budget.
    llm, responses = _llm({"answer": 1}, max_budget=0.5)

    first = asyncio.run(llm.structured_completion("first", {}))

    assert first == {"answer": 1}
    with pytest.raises(BudgetExceededError):
        asyncio.run(llm.structured_completion("second", {}))
    with pytest.raises(BudgetExceededError):
        asyncio.run(llm.web_search_completion("third"))
    assert len(responses.requests) == 1


def test_no_budget_means_no_cap():
    llm, responses = _llm(max_budget=None)

    asyncio.run(llm.structured_completion("first", {}))
    asyncio.run(llm.structured_completion("second", {}))

    assert len(responses.requests) == 2


class _FailingLLM:
    def __init__(self, error: Exception):
        self._error = error

    async def structured_completion(self, **_):
        raise self._error


def _classify(llm) -> list[dict]:
    return asyncio.run(
        scope_audit._classify_in_batches(
            llm,
            [{"label": "A"}, {"label": "B"}],
            prompt="{entities}",
            response_format={},
            variables={},
            parse=lambda brief, item: item,
            fallback=lambda label, reason: {"label": label, "reason": reason},
            batch_size=1,
            concurrency=1,
            progress_cb=None,
        )
    )


def test_audit_stops_when_the_budget_is_spent():
    with pytest.raises(BudgetExceededError):
        _classify(_FailingLLM(BudgetExceededError("over budget")))


def test_audit_still_keeps_records_on_an_ordinary_failure():
    results = _classify(_FailingLLM(RuntimeError("boom")))

    assert [r["label"] for r in results] == ["A", "B"]
    assert all("boom" in r["reason"] for r in results)


# --- prompt text ------------------------------------------------------------


def test_prompt_without_variables_is_sent_verbatim():
    llm, responses = _llm()
    prompt = 'Reply as {"name": "..."} for the label {Acme}'

    asyncio.run(llm.structured_completion(prompt, {}))
    asyncio.run(llm.web_search_completion(prompt))

    assert [r["input"] for r in responses.requests] == [prompt, prompt]


def test_prompt_with_variables_is_still_formatted():
    llm, responses = _llm()

    asyncio.run(llm.structured_completion("Hello {name} {{literal}}", {}, variables={"name": "Acme"}))

    assert responses.requests[0]["input"] == "Hello Acme {literal}"


# --- duplicate detection ----------------------------------------------------

NEAR_MISS = ("Contoso Case Tracker", "Contoso Case Manager")


def test_fuzzy_miss_reports_its_best_score():
    label, existing = NEAR_MISS

    match, score = find_fuzzy_match(label, [existing], threshold=85)

    assert match is None
    assert 65 < score < 85
    assert find_fuzzy_match("Zebra", ["Quantum Orchard"], threshold=85)[0] is None


class _DedupLLM:
    def __init__(self, duplicates: list[dict]):
        self.calls: list[dict] = []
        self._duplicates = duplicates

    async def structured_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"duplicates": self._duplicates}


def _check(llm, new_labels: list[str], existing: list[str]) -> dict[str, str]:
    engine = ResolutionEngine(_config(), llm)
    return asyncio.run(engine.check_duplicates(new_labels, existing))


def test_near_miss_labels_are_put_to_the_model():
    label, existing = NEAR_MISS
    llm = _DedupLLM([{"new_label": label, "existing_label": existing, "confidence": 0.95}])

    duplicates = _check(llm, [label], [existing])

    assert duplicates == {label: existing}
    assert llm.calls[0]["variables"]["new_labels"] == label


def test_unrelated_labels_are_not_put_to_the_model():
    llm = _DedupLLM([])

    assert _check(llm, ["Zebra"], ["Quantum Orchard"]) == {}
    assert llm.calls == []


def test_model_duplicates_naming_unknown_labels_are_ignored():
    label, existing = NEAR_MISS
    llm = _DedupLLM([
        {"new_label": label, "existing_label": "Never Seen", "confidence": 0.99},
        {"new_label": "Also Never Seen", "existing_label": existing, "confidence": 0.99},
    ])

    assert _check(llm, [label], [existing]) == {}


# --- completion retries -----------------------------------------------------


def _schemify_with_one_incomplete_record(tmp_path, monkeypatch, *, cache_enabled: bool):
    monkeypatch.chdir(tmp_path)  # the cache directory is created in the working directory
    schemify = Schemify(_config(cache_enabled=cache_enabled))
    schemify.record_set = RecordSet(
        category="tools",
        guidance="",
        records=[Record(label="Acme")],
        schema_attributes=[SchemaAttribute(name="Country"), SchemaAttribute(name="Founded")],
    )
    calls: list[list[str]] = []

    async def expand_record(record, record_set, target_attributes=None, **_):
        calls.append(list(target_attributes or []))
        return record

    monkeypatch.setattr(schemify.extraction, "expand_record", expand_record)
    return schemify, calls


def test_completion_does_not_retry_a_cached_page_that_filled_nothing(tmp_path, monkeypatch):
    schemify, calls = _schemify_with_one_incomplete_record(tmp_path, monkeypatch, cache_enabled=True)

    asyncio.run(schemify.complete_all(verbose=False, max_calls_per_entity=3))

    assert len(calls) == 1


def test_completion_still_retries_when_each_call_searches_afresh(tmp_path, monkeypatch):
    schemify, calls = _schemify_with_one_incomplete_record(tmp_path, monkeypatch, cache_enabled=False)

    asyncio.run(schemify.complete_all(verbose=False, max_calls_per_entity=3))

    assert len(calls) == 3
