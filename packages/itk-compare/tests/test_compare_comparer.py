# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from itk_compare import (
    AttributeFilter,
    CaseGroupComparer,
    CompareConfig,
    CompareResult,
    ReportSelection,
)
from itk_compare.report import build_report_messages, report_variables
from itk_core.classes import LLMCallback

if TYPE_CHECKING:
    from itk_compare_testing import FakeChat


def test_summarize_keeps_every_intermediate_frame(temporal_result: CompareResult):
    assert temporal_result.prepared.height == 7
    assert temporal_result.filtered.height == 7
    assert temporal_result.groups_frame.height == 3
    assert temporal_result.attributes_frame.height == 12
    assert temporal_result.temporal_frame is not None
    assert temporal_result.dataset_proportion == 100


def test_summarize_without_temporal_has_no_temporal_frame(result: CompareResult):
    assert result.temporal_frame is None
    assert "period_window" not in result.summary.columns


def test_filters_shrink_the_filtered_frame_and_the_proportion(
    complaints: pl.DataFrame,
):
    config = CompareConfig(
        groups=("city",),
        aggregates=("price_issue",),
        filters=(AttributeFilter(attribute="product", value="B"),),
    )
    result = CaseGroupComparer.summarize(complaints, config)
    assert result.filtered.height == 2
    assert result.dataset_proportion == 29
    assert sorted(result.summary["city"].unique().to_list()) == ["Hilltop", "Lakeside"]
    assert "**29%**" in result.description


def test_summarize_rejects_unknown_columns(complaints: pl.DataFrame):
    config = CompareConfig(groups=("city",), aggregates=("nope",))
    with pytest.raises(KeyError, match="nope"):
        CaseGroupComparer.summarize(complaints, config)


def test_summarize_of_an_empty_frame_gives_empty_frames():
    frame = pl.DataFrame(schema={"city": pl.Utf8, "issue": pl.Utf8, "period": pl.Utf8})
    config = CompareConfig(groups=("city",), aggregates=("issue",), temporal="period")
    result = CaseGroupComparer.summarize(frame, config)
    assert result.summary.height == 0
    assert result.summary.columns == config.summary_columns
    assert result.dataset_proportion == 0


def test_summarize_is_deterministic(complaints: pl.DataFrame, temporal_config):
    first = CaseGroupComparer.summarize(complaints, temporal_config).summary
    shuffled = complaints.sample(fraction=1.0, shuffle=True, seed=7)
    second = CaseGroupComparer.summarize(shuffled, temporal_config).summary
    assert first.equals(second)


def test_group_options_and_report_rows_come_from_the_result(result: CompareResult):
    assert CaseGroupComparer.group_options(result) == [
        {"city": "Hilltop"},
        {"city": "Lakeside"},
        {"city": "Seaside"},
    ]
    rows, text = CaseGroupComparer.report_rows(
        result, ReportSelection(top_group_ranks=2)
    )
    assert rows.height == 8
    assert text == "Filtered to the top 2 groups by record count"
    everything, _ = CaseGroupComparer.report_rows(result)
    assert everything.height == result.summary.height


def test_report_messages_carry_description_rows_and_filter_text(result: CompareResult):
    rows, text = CaseGroupComparer.report_rows(
        result, ReportSelection(top_group_ranks=2)
    )
    messages = build_report_messages(report_variables(result.description, rows, text))
    assert [m["role"] for m in messages] == ["system"]
    content = messages[0]["content"]
    assert "This table shows:" in content
    assert "Filtered to the top 2 groups by record count" in content
    assert rows.write_csv() in content
    assert "plain English" in content


def test_report_messages_accept_custom_instructions(result: CompareResult):
    messages = build_report_messages(
        report_variables(result.description, result.summary), "Answer in French."
    )
    assert "Answer in French." in messages[0]["content"]
    assert "plain English" not in messages[0]["content"]


def test_report_uses_the_chat_function_and_streams_to_callbacks(
    result: CompareResult, fake: FakeChat
):
    comparer = CaseGroupComparer(fake)
    callback = LLMCallback()
    text = comparer.report(
        result, ReportSelection(groups=({"city": "Seaside"},)), callbacks=[callback]
    )
    assert text == fake.answer
    assert len(fake.calls) == 1
    assert "Seaside" in fake.calls[0][0]["content"]
    assert "Lakeside,3,2" not in fake.calls[0][0]["content"]
    assert callback.response == [fake.answer]


def test_report_without_a_chat_function_is_an_error(result: CompareResult):
    with pytest.raises(ValueError, match="chat function"):
        CaseGroupComparer().report(result)
