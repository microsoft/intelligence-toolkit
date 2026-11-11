# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The one model call the package makes, behind a protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from itk_core.openai_configuration import OpenAIConfiguration


class ChatFn(Protocol):
    """One chat completion.

    ``response_schema`` is a JSON schema the reply must conform to, or ``None``
    for free text.
    """

    async def __call__(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str: ...


def response_format(response_schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict structured-output wrapper around a JSON schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_object",
            "strict": True,
            "schema": response_schema,
        },
    }


def openai_chat(configuration: OpenAIConfiguration | None = None) -> ChatFn:
    """Adapt itk-core's ``OpenAIClient`` to ``ChatFn``."""
    from itk_core.client import OpenAIClient

    client = OpenAIClient(configuration)

    async def chat(
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if response_schema is not None:
            kwargs["response_format"] = response_format(response_schema)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return await client.generate_chat_async(messages, stream=False, **kwargs)

    return chat
