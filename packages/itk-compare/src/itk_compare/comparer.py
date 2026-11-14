# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs the pipeline end to end and reports from the kept result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from itk_compare.groups import attribute_counts, group_counts
from itk_compare.models import CompareResult, ReportSelection
from itk_compare.prepare import apply_filters, check_columns, prepare_frame
from itk_compare.report import generate_report
from itk_compare.summary import build_summary, group_options, select_report_rows
from itk_compare.temporal import temporal_frame

if TYPE_CHECKING:
    import polars as pl
    from itk_core.classes import LLMCallback
    from itk_core.openai_configuration import OpenAIConfiguration

    from itk_compare.models import CompareConfig, GroupKey
    from itk_compare.report import ChatFn


class CaseGroupComparer:
    """Summarise case records by group; optionally hold a chat function for reports."""

    def __init__(self, chat: ChatFn | None = None) -> None:
        self.chat = chat

    @classmethod
    def from_openai(cls, configuration: OpenAIConfiguration) -> CaseGroupComparer:
        from itk_core.client import OpenAIClient

        return cls(OpenAIClient(configuration).generate_chat)

    @staticmethod
    def summarize(frame: pl.DataFrame, config: CompareConfig) -> CompareResult:
        """Prepare, filter, count groups and attributes, add windows, join the summary."""
        check_columns(frame, config)
        prepared = prepare_frame(frame, config.groups)
        filtered = apply_filters(prepared, config.filters)
        groups_frame = group_counts(filtered, config.groups)
        attributes_frame = attribute_counts(filtered, config.groups, config.aggregates)
        temporal = (
            temporal_frame(filtered, config.groups, config.temporal, config.aggregates)
            if config.temporal is not None
            else None
        )
        summary = build_summary(config, groups_frame, attributes_frame, temporal)
        return CompareResult(
            config=config,
            prepared=prepared,
            filtered=filtered,
            groups_frame=groups_frame,
            attributes_frame=attributes_frame,
            temporal_frame=temporal,
            summary=summary,
        )

    @staticmethod
    def group_options(result: CompareResult) -> list[GroupKey]:
        return group_options(result.summary, result.config.groups)

    @staticmethod
    def report_rows(
        result: CompareResult, selection: ReportSelection | None = None
    ) -> tuple[pl.DataFrame, str]:
        """Summary rows for a report and the sentence describing the selection."""
        return select_report_rows(result.summary, selection or ReportSelection())

    def report(
        self,
        result: CompareResult,
        selection: ReportSelection | None = None,
        instructions: str | None = None,
        callbacks: list[LLMCallback] | None = None,
    ) -> str:
        """Generate the markdown report for the selected rows."""
        if self.chat is None:
            msg = "CaseGroupComparer needs a chat function to generate reports"
            raise ValueError(msg)
        rows, filter_description = self.report_rows(result, selection)
        return generate_report(
            self.chat,
            result.description,
            rows,
            filter_description,
            instructions,
            callbacks,
        )
