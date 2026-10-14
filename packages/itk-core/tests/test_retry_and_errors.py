# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from itk_core.base_batch_async import BaseBatchAsync
from itk_core.base_chat import BaseChat
from itk_core.client import (
    OpenAIClient,
    _default_reasoning_effort,
    _is_reasoning_model,
)
from itk_core.errors import (
    LLMError,
    is_transient_error,
    retry_after_seconds,
)
from itk_core.openai_configuration import OpenAIConfiguration
from itk_core.utils import get_token_count, truncate_to_tokens

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status: int, headers: dict | None = None):
    response = httpx.Response(status, request=_REQUEST, headers=headers or {})
    return cls("error", response=response, body=None)


@pytest.fixture(autouse=True)
def no_backoff_delay():
    with patch("itk_core.retry._compute_delay", return_value=0):
        yield


def _client(model: str = "gpt-4"):
    config = OpenAIConfiguration({"api_key": "k", "model": model, "api_type": "OpenAI"})
    with patch("itk_core.client.OpenAI") as sync_cls, \
         patch("itk_core.client.AsyncOpenAI") as async_cls:
        sync_cls.return_value = MagicMock()
        async_cls.return_value = MagicMock()
        return OpenAIClient(config)


def _response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# --- transient error detection ---------------------------------------------

def test_transient_errors_detected():
    assert is_transient_error(_status_error(openai.RateLimitError, 429))
    assert is_transient_error(_status_error(openai.InternalServerError, 503))
    assert is_transient_error(openai.APIConnectionError(request=_REQUEST))
    assert is_transient_error(asyncio.TimeoutError())


def test_non_transient_errors_not_retried():
    assert not is_transient_error(_status_error(openai.AuthenticationError, 401))
    assert not is_transient_error(_status_error(openai.BadRequestError, 400))
    assert not is_transient_error(ValueError("bad input"))


def test_transient_error_detected_through_cause_chain():
    try:
        try:
            raise _status_error(openai.RateLimitError, 429)
        except Exception as inner:
            raise LLMError("wrapped") from inner
    except LLMError as outer:
        assert is_transient_error(outer)


def test_retry_after_header_is_honoured_and_capped():
    assert retry_after_seconds(_status_error(openai.RateLimitError, 429, {"retry-after": "3"})) == 3
    assert retry_after_seconds(_status_error(openai.RateLimitError, 429, {"retry-after-ms": "250"})) == 0.25
    assert retry_after_seconds(_status_error(openai.RateLimitError, 429, {"retry-after": "9999"})) == 60
    assert retry_after_seconds(_status_error(openai.RateLimitError, 429)) is None


# --- client retries --------------------------------------------------------

@pytest.mark.asyncio
async def test_async_chat_retries_rate_limit_then_succeeds():
    client = _client()
    client._async_client.chat.completions.create = AsyncMock(
        side_effect=[
            _status_error(openai.RateLimitError, 429),
            _status_error(openai.RateLimitError, 429),
            _response("ok"),
        ]
    )
    result = await client.generate_chat_async([{"role": "user", "content": "hi"}], stream=False)
    assert result == "ok"
    assert client._async_client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_async_chat_does_not_retry_auth_errors():
    client = _client()
    client._async_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.AuthenticationError, 401)
    )
    with pytest.raises(LLMError, match="Problem in OpenAI response"):
        await client.generate_chat_async([{"role": "user", "content": "hi"}], stream=False)
    assert client._async_client.chat.completions.create.await_count == 1


def test_sync_chat_retries_server_errors_then_gives_up():
    client = _client()
    client._client.chat.completions.create = MagicMock(
        side_effect=_status_error(openai.InternalServerError, 500)
    )
    with pytest.raises(LLMError):
        client.generate_chat([{"role": "user", "content": "hi"}], stream=False)
    # initial attempt + LLM_MAX_RETRIES
    assert client._client.chat.completions.create.call_count == 5


@pytest.mark.asyncio
async def test_async_embedding_retries_timeouts():
    client = _client()
    embedding = MagicMock()
    embedding.data = [MagicMock(embedding=[0.1])]
    client._async_client.embeddings.create = AsyncMock(
        side_effect=[openai.APITimeoutError(request=_REQUEST), embedding]
    )
    assert await client.generate_embedding_async("text") == [0.1]


# --- streaming -------------------------------------------------------------

def _chunk(content):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    return chunk


def _empty_chunk():
    chunk = MagicMock()
    chunk.choices = []
    return chunk


def test_streaming_without_callbacks_returns_text():
    client = _client()
    client._client.chat.completions.create = MagicMock(
        return_value=[_chunk("Hello"), _empty_chunk(), _chunk(" World"), _chunk(None)]
    )
    assert client.generate_chat([{"role": "user", "content": "hi"}]) == "Hello World"


@pytest.mark.asyncio
async def test_async_streaming_skips_chunks_without_choices():
    client = _client()

    async def stream():
        for chunk in [_empty_chunk(), _chunk("a"), _empty_chunk(), _chunk("b")]:
            yield chunk

    client._async_client.chat.completions.create = AsyncMock(return_value=stream())
    callback = MagicMock()
    result = await client.generate_chat_async(
        [{"role": "user", "content": "hi"}], stream=True, callbacks=[callback]
    )
    assert result == "ab"
    callback.on_llm_new_token.assert_called_with("ab▌")


# --- reasoning models ------------------------------------------------------

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.4-mini", "none"),
        ("gpt-5.1", "none"),
        ("gpt-5", "minimal"),
        ("gpt-5-mini", "minimal"),
        ("o3-mini", None),
        ("o4-mini", None),
    ],
)
def test_default_reasoning_effort(model, expected):
    assert _default_reasoning_effort(model) == expected


def test_chat_models_are_not_reasoning_models():
    assert not _is_reasoning_model("gpt-5-chat-latest")
    assert not _is_reasoning_model("gpt-4.1")
    assert _is_reasoning_model("o3")


def test_o_series_call_omits_reasoning_effort_and_temperature():
    client = _client("o3-mini")
    client._client.chat.completions.create = MagicMock(return_value=_response("ok"))
    client.generate_chat([{"role": "user", "content": "hi"}], stream=False, max_tokens=50)
    kwargs = client._client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["max_completion_tokens"] == 256


def test_reasoning_model_drops_logit_bias_and_floors_tiny_budget():
    client = _client("gpt-5.4")
    client._client.chat.completions.create = MagicMock(return_value=_response("Yes"))
    client.generate_chat(
        [{"role": "user", "content": "hi"}], stream=False, max_tokens=1, logit_bias={1: 5}
    )
    kwargs = client._client.chat.completions.create.call_args.kwargs
    assert "logit_bias" not in kwargs
    assert kwargs["max_completion_tokens"] == 256


def test_reasoning_model_keeps_large_budget():
    client = _client("gpt-5.4")
    client._client.chat.completions.create = MagicMock(return_value=_response("ok"))
    client.generate_chat([{"role": "user", "content": "hi"}], stream=False, max_tokens=4000)
    assert client._client.chat.completions.create.call_args.kwargs["max_completion_tokens"] == 4000


def test_chat_model_keeps_logit_bias_and_max_tokens():
    client = _client("gpt-4.1-mini")
    client._client.chat.completions.create = MagicMock(return_value=_response("Yes"))
    client.generate_chat(
        [{"role": "user", "content": "hi"}], stream=False, max_tokens=1, logit_bias={1: 5}
    )
    kwargs = client._client.chat.completions.create.call_args.kwargs
    assert kwargs["logit_bias"] == {1: 5}
    assert kwargs["max_tokens"] == 1


def test_explicit_reasoning_effort_is_preserved():
    client = _client("gpt-5.4-mini")
    client._client.chat.completions.create = MagicMock(return_value=_response("ok"))
    client.generate_chat(
        [{"role": "user", "content": "hi"}], stream=False, reasoning_effort="high"
    )
    assert client._client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "high"


# --- batching --------------------------------------------------------------

def test_semaphore_is_rebound_per_event_loop():
    batch = BaseBatchAsync()
    batch.set_concurrency(2)

    async def grab():
        sem = batch.semaphore
        async with sem:
            await asyncio.sleep(0)
        return sem

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second
    assert second._value == 2


@pytest.mark.asyncio
async def test_generate_texts_async_resets_progress_between_runs():
    config = OpenAIConfiguration({"api_key": "k", "model": "gpt-4", "api_type": "OpenAI"})
    with patch("itk_core.client.OpenAI"), \
         patch("itk_core.client.AsyncOpenAI"):
        chat = BaseChat(config)
    chat.generate_chat_async = AsyncMock(return_value="r")
    callback = MagicMock()

    await chat.generate_texts_async([[{"role": "user", "content": "a"}]] * 3, callbacks=[callback])
    assert chat.completed_tasks == 3
    await chat.generate_texts_async([[{"role": "user", "content": "a"}]] * 2, callbacks=[callback])
    assert (chat.completed_tasks, chat.total_tasks) == (2, 2)


@pytest.mark.asyncio
async def test_generate_texts_async_does_not_double_wrap_errors():
    config = OpenAIConfiguration({"api_key": "k", "model": "gpt-4", "api_type": "OpenAI"})
    with patch("itk_core.client.OpenAI"), \
         patch("itk_core.client.AsyncOpenAI"):
        chat = BaseChat(config)
    chat._async_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.BadRequestError, 400)
    )
    with pytest.raises(LLMError) as exc_info:
        await chat.generate_texts_async([[{"role": "user", "content": "a"}]])
    assert str(exc_info.value).count("Problem in OpenAI response") == 1


# --- token handling --------------------------------------------------------

def test_truncate_to_tokens_limits_token_count():
    text = "word " * 500
    truncated = truncate_to_tokens(text, 50)
    assert get_token_count(truncated) <= 50
    assert text.startswith(truncated)


def test_truncate_to_tokens_returns_short_text_unchanged():
    text = "short text"
    assert truncate_to_tokens(text, 50) is text


def test_async_client_rebuilt_per_event_loop():
    client = OpenAIClient(
        OpenAIConfiguration({"api_type": "OpenAI", "api_key": "test-key", "model": "gpt-4"})
    )

    async def grab():
        first = client._async_client
        assert client._async_client is first  # stable within one loop
        return first

    run1 = asyncio.run(grab())
    run2 = asyncio.run(grab())
    assert run1 is not run2
    assert client._async_client is run2  # outside a loop: last instance
