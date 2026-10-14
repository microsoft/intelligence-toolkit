# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Model interfaces used by the workflow, with adapters for itk_core clients.

Any object with matching ``complete``/``embed`` coroutines works, which keeps
the workflow testable without network access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itk_core.base_embedder import BaseEmbedder
    from itk_core.openai_configuration import OpenAIConfiguration

    from itk_query_text_data.models import TokenHandler

Message = dict[str, str]


@runtime_checkable
class LLM(Protocol):
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        logit_bias: dict[int, int] | None = None,
        on_token: TokenHandler | None = None,
    ) -> str:
        """Return the completion text.

        ``on_token`` streams the response: it receives the full text so far
        after each new token.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in order."""
        ...


class _TokenRelay:
    def __init__(self, on_token: TokenHandler) -> None:
        self._on_token = on_token

    def on_llm_new_token(self, text: str) -> None:
        self._on_token(text)


class OpenAILLM:
    """:class:`LLM` backed by :class:`itk_core.client.OpenAIClient` (OpenAI or Azure OpenAI)."""

    def __init__(self, configuration: OpenAIConfiguration) -> None:
        from itk_core.client import OpenAIClient

        self._client = OpenAIClient(configuration)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        logit_bias: dict[int, int] | None = None,
        on_token: TokenHandler | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if logit_bias is not None:
            kwargs["logit_bias"] = logit_bias
        return await self._client.generate_chat_async(
            list(messages),
            stream=on_token is not None,
            callbacks=None if on_token is None else [_TokenRelay(on_token)],
            **kwargs,
        )


class ItkEmbedder:
    """:class:`Embedder` backed by an :class:`itk_core.base_embedder.BaseEmbedder`.

    Texts longer than the embedder's token limit are truncated first.
    """

    def __init__(
        self, embedder: BaseEmbedder, callbacks: list[Any] | None = None
    ) -> None:
        self._embedder = embedder
        self._callbacks = callbacks

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = [{"hash": "", "text": text, "additional_details": {}} for text in texts]
        results = await self._embedder.embed_store_many(items, self._callbacks)
        return [list(item["vector"]) for item in results]
