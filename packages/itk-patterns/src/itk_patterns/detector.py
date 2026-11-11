# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs the pipeline end to end, one stage at a time."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl

from itk_patterns.detect import RecordCounter, detect_patterns
from itk_patterns.embed import EncoderEmbedder, fuse_embeddings
from itk_patterns.graph import period_graphs
from itk_patterns.models import (
    PAIR_SCHEMA,
    PATTERN,
    PATTERN_SCHEMA,
    TIME_SERIES_SCHEMA,
    DetectionConfig,
    PatternResult,
)
from itk_patterns.prepare import attribute_links, node_labels, prepare_frame
from itk_patterns.report import (
    build_report_messages,
    generate_report,
    report_variables,
)
from itk_patterns.series import attribute_counts, time_series

if TYPE_CHECKING:
    from itk_core.classes import LLMCallback
    from itk_core.openai_configuration import OpenAIConfiguration

    from itk_patterns.embed import PeriodEmbedder
    from itk_patterns.models import PatternConfig
    from itk_patterns.report import ChatFn


class PatternDetector:
    """Holds the engines; every stage takes a result and returns a new one."""

    def __init__(
        self, embedder: PeriodEmbedder | None = None, chat: ChatFn | None = None
    ) -> None:
        self.embedder: PeriodEmbedder = embedder or EncoderEmbedder()
        self.chat = chat

    @classmethod
    def from_openai(
        cls,
        configuration: OpenAIConfiguration,
        embedder: PeriodEmbedder | None = None,
    ) -> PatternDetector:
        from itk_core.client import OpenAIClient

        return cls(embedder, OpenAIClient(configuration).generate_chat_async)

    # -- stage 1 -------------------------------------------------------------

    @staticmethod
    def prepare(frame: pl.DataFrame, config: PatternConfig) -> PatternResult:
        """Cast the case table to text, link subjects to attributes, build period graphs."""
        source = prepare_frame(frame)
        links = attribute_links(source, config)
        return PatternResult(
            config=config,
            source=source,
            links=links,
            period_graphs=period_graphs(links, config),
            node_labels=node_labels(links),
        )

    # -- stage 2 -------------------------------------------------------------

    def embed(self, result: PatternResult) -> PatternResult:
        """Embed every period graph; positions and shifts land on the result."""
        positions, shifts = fuse_embeddings(
            result.period_graphs, result.node_labels, self.embedder, result.config
        )
        return replace(
            result,
            positions=positions,
            shifts=shifts,
            detection=None,
            converging_pairs=pl.DataFrame(schema=PAIR_SCHEMA),
            all_pairs=0,
            patterns=pl.DataFrame(schema=PATTERN_SCHEMA),
            time_series=pl.DataFrame(schema=TIME_SERIES_SCHEMA),
        )

    # -- stage 3 -------------------------------------------------------------

    @staticmethod
    def detect(
        result: PatternResult, detection: DetectionConfig | None = None
    ) -> PatternResult:
        """Find and score patterns, then their time series; no re-embedding."""
        if result.positions is None:
            msg = "PatternResult has not been embedded"
            raise ValueError(msg)
        detection = detection or DetectionConfig()
        counter = RecordCounter(result.links)
        patterns, pairs, all_pairs = detect_patterns(
            result.positions, result.links, result.config, detection, counter
        )
        series = time_series(
            result.links, patterns.get_column(PATTERN).to_list(), counter
        )
        return replace(
            result,
            detection=detection,
            converging_pairs=pairs,
            all_pairs=all_pairs,
            patterns=patterns,
            time_series=series,
        )

    # -- stage 4 -------------------------------------------------------------

    @staticmethod
    def attribute_counts(
        result: PatternResult, pattern: str, period: str
    ) -> pl.DataFrame:
        """Attribute value counts among the records matching ``pattern`` in ``period``."""
        return attribute_counts(result.source, pattern, period, result.config)

    @staticmethod
    def report_messages(
        result: PatternResult,
        pattern: str,
        period: str,
        instructions: str | None = None,
    ) -> list[dict[str, str]]:
        """The prompt the report is generated from, for inspection or a custom LLM."""
        variables = report_variables(
            pattern,
            period,
            result.pattern_series(pattern),
            PatternDetector.attribute_counts(result, pattern, period),
        )
        return build_report_messages(variables, instructions)

    async def report(
        self,
        result: PatternResult,
        pattern: str,
        period: str,
        instructions: str | None = None,
        callbacks: list[LLMCallback] | None = None,
    ) -> str:
        """Ask the LLM to explain one detected pattern."""
        if self.chat is None:
            msg = "PatternDetector has no chat function; pass one or use from_openai"
            raise ValueError(msg)
        messages = self.report_messages(result, pattern, period, instructions)
        return await generate_report(messages, self.chat, callbacks)
