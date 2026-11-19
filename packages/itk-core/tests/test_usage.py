# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from itk_core import usage as usage_module
from itk_core.client import OpenAIClient
from itk_core.openai_configuration import OpenAIConfiguration
from itk_core.usage import (
    LLMUsage,
    add_usage_callback,
    remove_usage_callback,
    usage_callback,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.fixture(autouse=True)
def no_leftover_callbacks():
    yield
    usage_module._callbacks.clear()


@pytest.fixture
def sdk():
    """An OpenAIClient with the sync and async SDK clients replaced by mocks."""
    config = OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "OpenAI",
    })
    with (
        patch("itk_core.client.OpenAI") as sync_class,
        patch("itk_core.client.AsyncOpenAI") as async_class,
    ):
        yield SimpleNamespace(
            client=OpenAIClient(config),
            sync=sync_class.return_value,
            asynchronous=async_class.return_value,
        )


def chat_response(text: str, usage: object) -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def chat_chunk(text: str | None, usage: object = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=text)
    choices = [] if text is None else [SimpleNamespace(delta=delta)]
    return SimpleNamespace(choices=choices, usage=usage)


def tokens(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def embedding_response() -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
    )


async def stream_of(chunks: list[SimpleNamespace]):
    for chunk in chunks:
        yield chunk


def test_chat_reports_usage(sdk):
    sdk.sync.chat.completions.create.return_value = chat_response("Hi", tokens(12, 3))
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        sdk.client.generate_chat(MESSAGES, stream=False)

    assert calls == [LLMUsage("chat", "gpt-4", 12, 3, 15)]


def test_streamed_chat_requests_and_reports_usage(sdk):
    sdk.sync.chat.completions.create.return_value = [
        chat_chunk("Hel"),
        chat_chunk("lo"),
        chat_chunk(None, tokens(12, 3)),
    ]
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        result = sdk.client.generate_chat(MESSAGES)

    assert result == "Hello"
    assert calls == [LLMUsage("chat", "gpt-4", 12, 3, 15)]
    sent = sdk.sync.chat.completions.create.call_args.kwargs
    assert sent["stream_options"] == {"include_usage": True}


def test_streamed_chat_keeps_caller_stream_options(sdk):
    sdk.sync.chat.completions.create.return_value = [chat_chunk("Hi")]

    with usage_callback(lambda _: None):
        sdk.client.generate_chat(MESSAGES, stream_options={"include_usage": False})

    sent = sdk.sync.chat.completions.create.call_args.kwargs
    assert sent["stream_options"] == {"include_usage": False}


def test_request_is_unchanged_without_callbacks(sdk):
    sdk.sync.chat.completions.create.return_value = [chat_chunk("Hi")]

    sdk.client.generate_chat(MESSAGES)

    assert "stream_options" not in sdk.sync.chat.completions.create.call_args.kwargs


def test_unreported_usage_is_none_not_zero(sdk):
    sdk.sync.chat.completions.create.return_value = [chat_chunk("Hi")]
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        sdk.client.generate_chat(MESSAGES)

    assert calls == [LLMUsage("chat", "gpt-4", None, None, None)]


async def test_async_chat_reports_usage(sdk):
    sdk.asynchronous.chat.completions.create = AsyncMock(
        return_value=chat_response("Hi", tokens(5, 2))
    )
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        await sdk.client.generate_chat_async(MESSAGES, stream=False)

    assert calls == [LLMUsage("chat", "gpt-4", 5, 2, 7)]


async def test_async_streamed_chat_reports_usage(sdk):
    sdk.asynchronous.chat.completions.create = AsyncMock(
        return_value=stream_of([chat_chunk("Hi"), chat_chunk(None, tokens(5, 2))])
    )
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        result = await sdk.client.generate_chat_async(MESSAGES)

    assert result == "Hi"
    assert calls == [LLMUsage("chat", "gpt-4", 5, 2, 7)]


def test_embeddings_report_usage(sdk):
    sdk.sync.embeddings.create.return_value = embedding_response()
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        sdk.client.generate_embedding("text", model="embed-model")
        sdk.client.generate_embeddings(["a", "b"], model="embed-model")

    assert calls == [LLMUsage("embedding", "embed-model", 7, None, 7)] * 2


async def test_async_embedding_reports_usage(sdk):
    sdk.asynchronous.embeddings.create = AsyncMock(return_value=embedding_response())
    calls: list[LLMUsage] = []

    with usage_callback(calls.append):
        await sdk.client.generate_embedding_async("text", model="embed-model")

    assert calls == [LLMUsage("embedding", "embed-model", 7, None, 7)]


def test_failing_callback_does_not_fail_the_call(sdk):
    sdk.sync.chat.completions.create.return_value = chat_response("Hi", tokens(1, 1))
    calls: list[LLMUsage] = []
    add_usage_callback(MagicMock(side_effect=RuntimeError("ledger down")))
    add_usage_callback(calls.append)

    result = sdk.client.generate_chat(MESSAGES, stream=False)

    assert result == "Hi"
    assert len(calls) == 1


def test_removed_callback_is_not_called(sdk):
    sdk.sync.chat.completions.create.return_value = chat_response("Hi", tokens(1, 1))
    calls: list[LLMUsage] = []
    add_usage_callback(calls.append)
    remove_usage_callback(calls.append)
    remove_usage_callback(calls.append)

    sdk.client.generate_chat(MESSAGES, stream=False)

    assert calls == []
