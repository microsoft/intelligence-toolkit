# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs the pipeline end to end."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from itk_extract.chunk import chunk_sources, token_splitter
from itk_extract.extract import extract_chunks
from itk_extract.merge import merge
from itk_extract.models import ExtractConfig, ExtractResult
from itk_extract.records import to_frames
from itk_extract.schema import RecordSchema
from itk_extract.sources import check_unique_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from itk_core.openai_configuration import OpenAIConfiguration
    from itk_core.progress import ProgressBatchCallback

    from itk_extract.chunk import Splitter
    from itk_extract.extract import BatchChatFn
    from itk_extract.models import Source


class Extractor:
    """Holds the chat function; everything else is passed in and returned."""

    def __init__(
        self,
        chat: BatchChatFn,
        splitter_factory: Callable[[int], Splitter] = token_splitter,
    ) -> None:
        self.chat = chat
        self.splitter_factory = splitter_factory

    @classmethod
    def from_openai(
        cls, configuration: OpenAIConfiguration, concurrency: int | None = None
    ) -> Extractor:
        from itk_core.base_chat import BaseChat

        return cls(BaseChat(configuration, concurrency).generate_texts_async)

    async def extract(
        self,
        sources: list[Source],
        schema: RecordSchema | dict[str, Any],
        config: ExtractConfig | None = None,
        callbacks: list[ProgressBatchCallback] | None = None,
    ) -> ExtractResult:
        """Chunk, extract, merge and tabulate the given sources."""
        config = config or ExtractConfig()
        record_schema = (
            schema if isinstance(schema, RecordSchema) else RecordSchema(schema)
        )
        check_unique_ids(sources)
        if config.max_sources is not None:
            sources = sources[: config.max_sources]
        splitter = (
            self.splitter_factory(config.chunk_tokens) if config.chunk_tokens else None
        )
        chunks = chunk_sources(sources, splitter)
        extractions, errors = await extract_chunks(
            chunks, self.chat, record_schema, config, callbacks
        )
        return self.remerge(
            ExtractResult(record_schema, config, extractions, errors, {}, []), config
        )

    @staticmethod
    def remerge(result: ExtractResult, config: ExtractConfig) -> ExtractResult:
        """Re-merge and re-tabulate with a new config, without calling the model."""
        merged, conflicts = merge(
            result.extractions, config.merge_policy, config.annotate_provenance
        )
        frames = to_frames(merged, result.schema, config.dedupe_records)
        return ExtractResult(
            result.schema,
            config,
            result.extractions,
            result.errors,
            merged,
            conflicts,
            frames,
        )
