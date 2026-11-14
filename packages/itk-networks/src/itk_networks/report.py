# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build the report prompt from a simplified network and ask the LLM for it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from itk_core.utils import generate_messages

from itk_networks import prompts

if TYPE_CHECKING:
    import polars as pl
    from itk_core.classes import LLMCallback


class ChatFn(Protocol):
    """Matches OpenAIClient.generate_chat_async."""

    async def __call__(
        self,
        messages: list[dict[str, str]],
        stream: bool = ...,
        callbacks: list[LLMCallback] | None = ...,
    ) -> str: ...


def report_variables(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    network_id: int,
    entity: str = "",
    exposure_markdown: str = "",
    max_flags: int = 0,
    mean_flags: float = 0.0,
) -> dict[str, Any]:
    """The exact variables the legacy ``generate_report`` filled the prompt with."""
    return {
        "entity_id": entity,
        "network_id": network_id,
        "max_flags": max_flags,
        "mean_flags": mean_flags,
        "exposure": exposure_markdown,
        "network_nodes": nodes.write_csv(),
        "network_edges": edges.write_csv(),
    }


def build_report_messages(
    variables: dict[str, Any], instructions: str | None = None
) -> list[dict[str, str]]:
    return generate_messages(
        instructions if instructions is not None else prompts.USER_PROMPT,
        prompts.REPORT_PROMPT,
        variables,
        prompts.SAFETY_PROMPT,
    )


async def generate_report(
    messages: list[dict[str, str]],
    chat: ChatFn,
    callbacks: list[LLMCallback] | None = None,
) -> str:
    return await chat(messages, stream=False, callbacks=callbacks)
