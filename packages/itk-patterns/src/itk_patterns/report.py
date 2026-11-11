# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build the pattern report prompt and ask the LLM for the report."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from itk_core.utils import generate_messages

from itk_patterns import prompts

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
    pattern: str,
    period: str,
    time_series: pl.DataFrame,
    attribute_counts: pl.DataFrame,
) -> dict[str, Any]:
    """The variables the legacy ``explain_pattern`` filled the prompt with."""
    return {
        "pattern": pattern,
        "period": period,
        "time_series": time_series.write_csv(),
        "attribute_counts": attribute_counts.write_csv(),
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


__all__ = ["ChatFn", "build_report_messages", "generate_report", "report_variables"]
