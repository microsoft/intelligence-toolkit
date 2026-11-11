# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
import pytest
from itk_compare import AttributeFilter, CompareConfig, CompareResult, ReportSelection
from itk_compare.summary import (
    build_summary,
    describe_summary,
    group_options,
    select_report_rows,
)
from itk_compare_testing import rows


def test_summary_has_legacy_columns_and_one_row_per_group_and_value(
    result: CompareResult,
):
    assert result.summary.columns == result.config.summary_columns
    assert result.summary.height == 3 * 4
    assert (
        rows(result.summary, "city", "group_count", "group_rank")[:4]
        == [("Hilltop", 3, 2)] * 4
    )


def test_temporal_summary_adds_window_columns_and_sorts_by_window(
    temporal_result: CompareResult,
):
    summary = temporal_result.summary
    assert summary.columns[-4:] == [
        "period_window",
        "period_window_count",
        "period_window_rank",
        "period_window_delta",
    ]
    keys = rows(summary, "city", "attribute_value", "period_window")
    assert keys == sorted(keys)
    # Seaside never shows price_issue:True in any window, so no window rows,
    # but the attribute frame still knows the overall zero count.
    assert (
        summary.filter(
            (pl.col("city") == "Seaside")
            & (pl.col("attribute_value") == "price_issue:True")
        ).height
        == 0
    )
    assert temporal_result.attributes_frame.filter(
        (pl.col("city") == "Seaside")
        & (pl.col("attribute_value") == "price_issue:True")
    )["group_attribute_count"].to_list() == [0]


def test_summary_counts_and_ranks_are_integers(temporal_result: CompareResult):
    schema = temporal_result.summary.schema
    for column in (
        "group_count",
        "group_rank",
        "group_attribute_count",
        "group_attribute_rank",
        "period_window_count",
        "period_window_rank",
        "period_window_delta",
    ):
        assert schema[column] == pl.Int64, column


def test_build_summary_requires_a_temporal_frame_when_configured(
    result: CompareResult,
):
    config = CompareConfig(groups=("city",), aggregates=("x",), temporal="period")
    with pytest.raises(ValueError, match="temporal frame"):
        build_summary(config, result.groups_frame, result.attributes_frame, None)


def test_describe_summary_without_filters():
    config = CompareConfig(groups=("city", "age"), aggregates=("issue",))
    text = describe_summary(config, 42, 100)
    assert text.splitlines() == [
        "This table shows:",
        "- A summary of all **42** data records with values for all grouping attributes",
        "- The **group_count** of records for all [**city**, **age**] groups, and "
        "corresponding **group_rank**",
        "- The **group_attribute_count** of each **attribute_value** for all "
        "[**city**, **age**] groups, and corresponding **group_attribute_rank**",
    ]


def test_describe_summary_with_filters_and_windows():
    config = CompareConfig(
        groups=("city",),
        aggregates=("issue",),
        temporal="period",
        filters=(AttributeFilter(attribute="product", value="A"),),
    )
    lines = describe_summary(config, 10, 25).splitlines()
    assert lines[1] == (
        "- A summary of **10** data records matching [**product\\:A**], representing "
        "**25%** of the overall dataset with values for all grouping attributes"
    )
    assert lines[4].startswith(
        "- The **period_window_count** of each **attribute_value**"
    )
    assert lines[5].startswith("- The **period_window_delta**, or change in")
    assert len(lines) == 6


def test_group_options_are_sorted_dicts(result: CompareResult):
    assert group_options(result.summary, ["city"]) == [
        {"city": "Hilltop"},
        {"city": "Lakeside"},
        {"city": "Seaside"},
    ]


def test_select_report_rows_by_named_groups(result: CompareResult):
    rows_, text = select_report_rows(
        result.summary, ReportSelection(groups=({"city": "Seaside"},))
    )
    assert rows_["city"].unique().to_list() == ["Seaside"]
    assert text == "Filtered to the following groups only: {'city': 'Seaside'}"


def test_select_report_rows_by_top_ranks_and_default(result: CompareResult):
    rows_, text = select_report_rows(result.summary, ReportSelection(top_group_ranks=2))
    assert sorted(rows_["city"].unique().to_list()) == ["Hilltop", "Lakeside"]
    assert text == "Filtered to the top 2 groups by record count"
    everything, no_text = select_report_rows(result.summary, ReportSelection())
    assert everything.height == result.summary.height
    assert no_text == ""


def test_named_groups_win_over_top_ranks(result: CompareResult):
    rows_, text = select_report_rows(
        result.summary,
        ReportSelection(groups=({"city": "Seaside"},), top_group_ranks=1),
    )
    assert rows_["city"].unique().to_list() == ["Seaside"]
    assert text.startswith("Filtered to the following groups only")
