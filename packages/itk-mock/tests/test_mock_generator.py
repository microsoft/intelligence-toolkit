# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest
from itk_mock import (
    MockConfig,
    MockDataError,
    MockDataGenerator,
    TextConfig,
)
from itk_mock.chat import openai_chat, response_format
from itk_mock.prompts import seeded_messages, text_messages, unseeded_messages
from pydantic import ValidationError

if TYPE_CHECKING:
    from itk_mock_testing import FakeChat


async def test_records_then_texts_without_regenerating_records(
    fake_chat: FakeChat, people_schema: dict[str, Any]
) -> None:
    generator = MockDataGenerator(fake_chat)
    config = MockConfig(num_records_overall=6, records_per_batch=3, seed=1)

    result = await generator.generate_records(people_schema, config)
    record_calls = len(fake_chat.calls)
    texts = await generator.generate_texts(
        result.primary_frame.select("name", "age"), TextConfig(parallelism=4)
    )

    assert record_calls == 2
    assert len(fake_chat.calls) == record_calls + 6
    assert texts.frame.height == 6
    assert texts.input_texts[0] == '{"name":"unseeded/0","age":20}'
    assert texts.texts[0] == 'Text about {"name":"unseeded/0","age":20}'


async def test_texts_from_plain_records(fake_chat: FakeChat) -> None:
    generator = MockDataGenerator(fake_chat)

    texts = await generator.generate_texts([{"name": "Ana"}, {"name": "Bo"}])

    assert texts.texts == ['Text about {"name":"Ana"}', 'Text about {"name":"Bo"}']


async def test_invalid_schema_is_rejected_before_any_call(
    fake_chat: FakeChat,
) -> None:
    schema = {"type": "object", "properties": {"rows": {"type": "nonsense"}}}

    with pytest.raises(MockDataError, match="not a valid JSON Schema"):
        await MockDataGenerator(fake_chat).generate_records(schema)
    assert fake_chat.calls == []


def test_result_and_configs_are_frozen() -> None:
    config = MockConfig()

    with pytest.raises(ValidationError):
        config.temperature = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TextConfig().parallelism = 2  # type: ignore[misc]
    assert dataclasses.is_dataclass(MockConfig) is False


def test_config_defaults_match_the_workflow() -> None:
    config = MockConfig()

    assert (
        config.num_records_overall,
        config.records_per_batch,
        config.duplicate_records_per_batch,
        config.related_records_per_batch,
        config.parallel_batches,
        config.temperature,
        config.seed,
    ) == (100, 10, 0, 0, 50, 0.5, None)
    assert (TextConfig().parallelism, TextConfig().temperature) == (10, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_records_overall": 5, "records_per_batch": 10},
        {"records_per_batch": 0},
        {"duplicate_records_per_batch": 6, "related_records_per_batch": 5},
        {"duplicate_records_per_batch": -1},
        {"parallel_batches": 0},
        {"temperature": 1.5},
    ],
)
def test_config_rejects_inconsistent_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MockConfig(**kwargs)


def test_prompts_are_single_system_messages() -> None:
    unseeded = unseeded_messages("Guidance {with braces}", ["records"], 7)
    seeded = seeded_messages({"name": "Ana"}, "", ["records"], 7, 1, 2)
    text = text_messages('{"name":"Ana"}', "Formal")

    for messages in (unseeded, seeded, text):
        assert [m["role"] for m in messages] == ["system"]
    assert "Guidance {with braces}" in unseeded[0]["content"]
    assert "Output record count: 7" in unseeded[0]["content"]
    assert "{'name': 'Ana'}" in seeded[0]["content"]
    assert (
        "Total records: 7\nNear duplicates of seed: 1\nClose relations of seed: 2"
        in seeded[0]["content"]
    )
    assert '{"name":"Ana"}' in text[0]["content"]
    assert "Formal" in text[0]["content"]


def test_response_format_is_strict_json_schema() -> None:
    schema = {"type": "object"}

    assert response_format(schema) == {
        "type": "json_schema",
        "json_schema": {"name": "answer_object", "strict": True, "schema": schema},
    }


async def test_openai_adapter_forwards_schema_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    class StubClient:
        def __init__(self, configuration: Any = None) -> None:
            self.configuration = configuration

        async def generate_chat_async(
            self, messages: Any, stream: bool = True, **kwargs: Any
        ) -> str:
            seen.append({"messages": messages, "stream": stream, **kwargs})
            return "reply"

    monkeypatch.setattr("itk_core.client.OpenAIClient", StubClient)
    chat = openai_chat()
    messages = [{"role": "system", "content": "hi"}]
    schema = {"type": "object"}

    assert await chat(messages, response_schema=schema, temperature=0.3) == "reply"
    assert await chat(messages) == "reply"

    assert seen[0] == {
        "messages": messages,
        "stream": False,
        "response_format": response_format(schema),
        "temperature": 0.3,
    }
    assert seen[1] == {"messages": messages, "stream": False}
    assert isinstance(MockDataGenerator.from_openai(), MockDataGenerator)
