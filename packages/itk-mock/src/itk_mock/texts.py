# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Generate one narrative text per record."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import polars as pl

from itk_mock.models import TextConfig, TextResult, text_frame
from itk_mock.prompts import text_messages

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from itk_mock.chat import ChatFn


def record_to_input_text(record: Mapping[str, Any]) -> str:
    """Compact JSON for one record, as it is shown to the model."""
    return json.dumps(dict(record), separators=(",", ":"), default=str)


def records_to_input_texts(
    records: pl.DataFrame | Sequence[Mapping[str, Any]],
) -> list[str]:
    """One input text per frame row or record."""
    rows = (
        records.iter_rows(named=True) if isinstance(records, pl.DataFrame) else records
    )
    return [record_to_input_text(row) for row in rows]


async def generate_texts(
    chat: ChatFn,
    input_texts: Sequence[str],
    config: TextConfig | None = None,
    on_update: Callable[[pl.DataFrame], None] | None = None,
) -> TextResult:
    """One text per input, ``config.parallelism`` at a time, in input order.

    ``on_update`` receives the ``mock_text`` frame after every batch.
    """
    config = config or TextConfig()
    texts: list[str] = []
    for start in range(0, len(input_texts), config.parallelism):
        batch = input_texts[start : start + config.parallelism]
        calls = [
            chat(
                text_messages(text, config.generation_guidance),
                temperature=config.temperature,
            )
            for text in batch
        ]
        texts.extend(await asyncio.gather(*calls))
        if on_update is not None:
            on_update(text_frame(texts))
    return TextResult(config=config, input_texts=list(input_texts), texts=texts)
