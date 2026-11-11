# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build the report prompt from selected summary rows and ask the LLM for it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from itk_core.utils import generate_messages

from itk_compare import prompts

if TYPE_CHECKING:
    import polars as pl
    from itk_core.classes import LLMCallback


class ChatFn(Protocol):
    """Matches ``OpenAIClient.generate_chat`` (synchronous, optionally streaming)."""

    def __call__(
        self,
        messages: list[dict[str, str]],
        callbacks: list[LLMCallback] | None = ...,
    ) -> str: ...


def report_variables(
    description: str, rows: pl.DataFrame, filter_description: str = ""
) -> dict[str, Any]:
    """The exact variables the legacy ``generate_group_report`` filled the prompt with."""
    return {
        "description": description,
        "dataset": rows.write_csv(),
        "filters": filter_description,
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


def generate_report(
    chat: ChatFn,
    description: str,
    rows: pl.DataFrame,
    filter_description: str = "",
    instructions: str | None = None,
    callbacks: list[LLMCallback] | None = None,
) -> str:
    """Ask ``chat`` for the markdown group comparison report."""
    messages = build_report_messages(
        report_variables(description, rows, filter_description), instructions
    )
    return chat(messages, callbacks=callbacks)
