# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs record and text generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from itk_mock.models import MockDataError, ValidationResult
from itk_mock.records import generate_records
from itk_mock.template import evaluate_schema
from itk_mock.texts import generate_texts, records_to_input_texts

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Mapping, Sequence

    import polars as pl
    from itk_core.openai_configuration import OpenAIConfiguration

    from itk_mock.chat import ChatFn
    from itk_mock.models import MockConfig, MockResult, TextConfig, TextResult


class MockDataGenerator:
    """Holds the chat function; everything else is passed in and returned."""

    def __init__(self, chat: ChatFn) -> None:
        self.chat = chat

    @classmethod
    def from_openai(
        cls, configuration: OpenAIConfiguration | None = None
    ) -> MockDataGenerator:
        from itk_mock.chat import openai_chat

        return cls(openai_chat(configuration))

    async def generate_records(
        self,
        schema: dict[str, Any],
        config: MockConfig | None = None,
        rng: random.Random | None = None,
        on_update: Callable[[dict[str, pl.DataFrame]], None] | None = None,
    ) -> MockResult:
        """Generate records for the schema; frames are kept on the result."""
        if evaluate_schema(schema) == ValidationResult.SCHEMA_INVALID:
            msg = "Schema is not a valid JSON Schema"
            raise MockDataError(msg)
        return await generate_records(self.chat, schema, config, rng, on_update)

    async def generate_texts(
        self,
        records: pl.DataFrame | Sequence[Mapping[str, Any]],
        config: TextConfig | None = None,
        on_update: Callable[[pl.DataFrame], None] | None = None,
    ) -> TextResult:
        """One text per frame row or record, without regenerating records."""
        return await generate_texts(
            self.chat, records_to_input_texts(records), config, on_update
        )
