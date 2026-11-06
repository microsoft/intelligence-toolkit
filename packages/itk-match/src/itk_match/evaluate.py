# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Ask the LLM how related the records in each group really are."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Protocol

import polars as pl
from itk_core.metaprompts import do_not_disrespect_context, do_not_harm
from pydantic import ValidationError

from itk_match.models import GROUP_ID, EvaluationBatch, GroupEvaluation

if TYPE_CHECKING:
    from itk_core.classes import LLMCallback

SYSTEM_PROMPT = """\
Goal: evaluate the overall RELATEDNESS of the records in each record group on a \
scale of 0-10, where 0 is definitively different entities and 10 is definitively \
the same entity or entity group (e.g., branches of a company).

Factors indicating unrelatedness: multiple fields having values that are different \
across grouped records, have no similarity, and are unrelated in the real world.
Factors indicating relatedness: multiple fields having values that are the same or \
similar across multiple grouped records, and are related in the real world.
Ignore inconsistent spelling, formatting, and missing values. The more additional \
fields that agree between records with similar names, the more likely they are related.
If names are in a language other than English, consider whether the English \
translations are generic descriptive terms (less likely related) or distinctive \
(more likely related). Keep explanations short and simple.

Respond with JSON only, in exactly this shape and with one entry per group_id:
{"evaluations": [{"group_id": 0, "relatedness": 7, "explanation": "..."}]}
"""

USER_PROMPT = """\
Record groups (CSV):

{data}
"""

SAFETY_PROMPT = f"{do_not_harm} {do_not_disrespect_context}"

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ChatFn(Protocol):
    """Matches OpenAIClient.generate_chat_async."""

    async def __call__(
        self,
        messages: list[dict[str, str]],
        stream: bool = ...,
        callbacks: list[LLMCallback] | None = ...,
    ) -> str: ...


def build_messages(groups_csv: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{SAFETY_PROMPT}"},
        {"role": "user", "content": USER_PROMPT.format(data=groups_csv)},
    ]


def parse_evaluations(text: str) -> list[GroupEvaluation]:
    """Parse the model's JSON reply, tolerating a code fence around it."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return EvaluationBatch.model_validate(json.loads(cleaned)).evaluations
    except (json.JSONDecodeError, ValidationError) as e:
        msg = f"Could not parse group evaluations: {e}"
        raise ValueError(msg) from e


def batch_groups(groups: pl.DataFrame, groups_per_batch: int) -> list[pl.DataFrame]:
    ids = groups.get_column(GROUP_ID).unique(maintain_order=True).to_list()
    return [
        groups.filter(pl.col(GROUP_ID).is_in(ids[i : i + groups_per_batch]))
        for i in range(0, len(ids), groups_per_batch)
    ]


def evaluations_frame(evaluations: list[GroupEvaluation]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            GROUP_ID: [e.group_id for e in evaluations],
            "relatedness": [e.relatedness for e in evaluations],
            "explanation": [e.explanation for e in evaluations],
        },
        schema={GROUP_ID: pl.UInt32, "relatedness": pl.UInt8, "explanation": pl.Utf8},
    )


async def evaluate_groups(
    groups: pl.DataFrame,
    chat: ChatFn,
    groups_per_batch: int = 20,
    callbacks: list[LLMCallback] | None = None,
) -> pl.DataFrame:
    """One relatedness row per group_id present in ``groups``.

    Groups are sent in batches; each reply must be JSON (see SYSTEM_PROMPT).
    """
    if groups.is_empty():
        return evaluations_frame([])
    evaluations: list[GroupEvaluation] = []
    for batch in batch_groups(groups, groups_per_batch):
        reply = await chat(
            build_messages(batch.write_csv()), stream=False, callbacks=callbacks
        )
        evaluations.extend(parse_evaluations(reply))
    return (
        evaluations_frame(evaluations)
        .unique(subset=[GROUP_ID], keep="last")
        .sort(GROUP_ID)
    )
