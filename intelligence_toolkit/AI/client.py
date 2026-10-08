# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import logging
import re
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from intelligence_toolkit.AI.classes import LLMCallback

from .defaults import (
    API_BASE_REQUIRED_FOR_AZURE,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT_SECONDS,
)
from .errors import EmbeddingError, LLMError, retry_transient
from .openai_configuration import OpenAIConfiguration

log = logging.getLogger(__name__)

STREAM_CURSOR = "▌"

_GPT5_VERSION = re.compile(r"^gpt-5(?:\.(\d+))?")


def _is_reasoning_model(model: str) -> bool:
    """Return True for models that accept the `reasoning_effort` parameter."""
    if not model:
        return False
    m = model.lower()
    if "chat" in m:
        return False
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _default_reasoning_effort(model: str) -> str | None:
    """Lowest reasoning effort the model accepts, or None to use the API default.

    gpt-5.1+ accepts "none"; the original gpt-5 family's lowest is "minimal";
    o-series models only accept low/medium/high, so we leave their default.
    """
    match = _GPT5_VERSION.match(model.lower())
    if match is None:
        return None
    minor = match.group(1)
    return "none" if minor is not None and int(minor) >= 1 else "minimal"


class OpenAIClient:
    """OpenAI Client class definition.

    Transient failures (rate limits, timeouts, connection and 5xx errors) are
    retried here with exponential backoff, honouring Retry-After. The SDK's own
    retry loop is disabled so there is a single retry policy.
    """

    _client = None
    _async_client = None

    def __init__(self, configuration: OpenAIConfiguration | None = None) -> None:
        self.configuration = configuration or OpenAIConfiguration()
        self._create_openai_client()

    def _create_openai_client(self) -> None:
        """Create a new OpenAI client instance."""
        if self.configuration.api_type == "Azure OpenAI":
            api_base = self.configuration.api_base
            if api_base is None:
                raise ValueError(API_BASE_REQUIRED_FOR_AZURE)
            log.info("Creating Azure OpenAI client api_base=%s", api_base)

            if self.configuration.az_auth_type == "Managed Identity":
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                auth = {"azure_ad_token_provider": token_provider}
            else:
                auth = {"api_key": self.configuration.api_key}

            common = {
                "api_version": self.configuration.api_version,
                "azure_endpoint": api_base,
                "max_retries": 0,
                **auth,
            }
            self._client = AzureOpenAI(**common)
            self._async_client = AsyncAzureOpenAI(**common)
        else:
            log.info("Creating OpenAI client")
            self._client = OpenAI(api_key=self.configuration.api_key, max_retries=0)
            self._async_client = AsyncOpenAI(
                api_key=self.configuration.api_key, max_retries=0
            )

    def _build_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        max_tokens = call_kwargs.pop("max_tokens", self.configuration.max_tokens)
        temperature = call_kwargs.pop("temperature", self.configuration.temperature)
        model = self.configuration.model
        if _is_reasoning_model(model):
            call_kwargs["max_completion_tokens"] = max_tokens
            if "reasoning_effort" not in call_kwargs:
                effort = _default_reasoning_effort(model)
                if effort is not None:
                    call_kwargs["reasoning_effort"] = effort
            # Reasoning models only support the default temperature.
        else:
            call_kwargs["max_tokens"] = max_tokens
            call_kwargs["temperature"] = temperature
        return call_kwargs

    @staticmethod
    def _chunk_delta(chunk) -> str:
        # Azure can emit chunks with no choices (e.g. content-filter results).
        if not chunk.choices:
            return ""
        return chunk.choices[0].delta.content or ""

    @staticmethod
    def _notify(callbacks: list[LLMCallback] | None, text: str, delta: str) -> None:
        if not callbacks:
            return
        show = text + STREAM_CURSOR if delta else text
        for callback in callbacks:
            callback.on_llm_new_token(show)

    @staticmethod
    def _message_content(response) -> str:
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    @retry_transient()
    def generate_chat(
        self,
        messages: list[str],
        stream: bool = True,
        callbacks: list[LLMCallback] | None = None,
        **kwargs,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.configuration.model,
                messages=messages,
                stream=stream,
                **self._build_call_kwargs(kwargs),
            )
            if not stream:
                return self._message_content(response)

            full_response = ""
            for chunk in response:
                delta = self._chunk_delta(chunk)
                full_response += delta
                self._notify(callbacks, full_response, delta)
            return full_response
        except Exception as e:
            log.warning("Chat completion failed: %s", e)
            msg = f"Problem in OpenAI response. {e}"
            raise LLMError(msg) from e

    @retry_transient()
    async def generate_chat_async(
        self,
        messages: list[str],
        stream: bool = True,
        callbacks: list[LLMCallback] | None = None,
        **kwargs,
    ) -> str:
        try:
            response = await self._async_client.chat.completions.create(
                model=self.configuration.model,
                messages=messages,
                stream=stream,
                **self._build_call_kwargs(kwargs),
            )
            if not stream:
                return self._message_content(response)

            full_response = ""
            async for chunk in response:
                delta = self._chunk_delta(chunk)
                full_response += delta
                self._notify(callbacks, full_response, delta)
            return full_response
        except Exception as e:
            log.warning("Async chat completion failed: %s", e)
            msg = f"Problem in OpenAI response. {e}"
            raise LLMError(msg) from e

    @retry_transient()
    def generate_embedding(
        self, text: str, model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        try:
            embedding = self._client.embeddings.create(
                input=text, model=model, timeout=EMBEDDING_TIMEOUT_SECONDS
            )
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise EmbeddingError(msg) from e
        return embedding.data[0].embedding

    @retry_transient()
    def generate_embeddings(self, text: list[str], model: str = DEFAULT_EMBEDDING_MODEL):
        try:
            return self._client.embeddings.create(
                input=text, model=model, timeout=EMBEDDING_TIMEOUT_SECONDS
            )
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise EmbeddingError(msg) from e

    @retry_transient()
    async def generate_embedding_async(
        self, text: str, model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        try:
            embedding = await self._async_client.embeddings.create(
                input=text, model=model, timeout=EMBEDDING_TIMEOUT_SECONDS
            )
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise EmbeddingError(msg) from e
        return embedding.data[0].embedding
