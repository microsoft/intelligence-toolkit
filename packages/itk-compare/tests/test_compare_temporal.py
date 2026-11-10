# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
from itk_compare.prepare import prepare_frame
from itk_compare.temporal import (
    complete_windows,
    temporal_frame,
    window_counts,
    window_deltas,
    window_ranks,
)
from itk_compare_testing import rows

AGGREGATES = ["price_issue", "quality_issue"]


def test_window_counts_skip_records_without_a_window():
    frame = pl.DataFrame({
        "g": ["a", "a", "a"],
        "t": ["1", "1", None],
        "v": ["x", "x", "x"],
    })
    counts = window_counts(frame, ["g"], "t", ["v"])
    assert counts.columns == ["g", "t_window", "attribute_value", "t_window_count"]
    assert rows(counts, "g", "t_window", "attribute_value", "t_window_count") == [
        ("a", "1", "v:x", 2)
    ]


def test_complete_windows_fills_zeros_and_drops_values_a_group_never_shows(
    complaints: pl.DataFrame,
):
    prepared = prepare_frame(complaints, ["city"])
    counts = window_counts(prepared, ["city"], "period", AGGREGATES)
    complete = complete_windows(counts, ["city"], "period", group_keys=prepared)
    seaside = complete.filter(pl.col("city") == "Seaside")
    # Seaside only ever shows the two False values; both windows are present.
    assert rows(seaside, "period_window", "attribute_value", "period_window_count") == [
        ("2023-H1", "price_issue:False", 0),
        ("2023-H1", "quality_issue:False", 0),
        ("2023-H2", "price_issue:False", 1),
        ("2023-H2", "quality_issue:False", 1),
    ]
    assert "price_issue:True" not in seaside["attribute_value"].to_list()


def test_complete_windows_of_empty_counts_is_empty():
    empty = pl.DataFrame(
        schema={"g": pl.Utf8, "t_window": pl.Utf8, "attribute_value": pl.Utf8}
    ).with_columns(pl.lit(0, dtype=pl.Int64).alias("t_window_count"))
    assert complete_windows(empty, ["g"], "t").height == 0


def test_window_deltas_difference_successive_windows_within_a_series():
    complete = pl.DataFrame({
        "g": ["a", "a", "a", "b", "b"],
        "t_window": ["1", "2", "3", "1", "2"],
        "attribute_value": ["v:x"] * 5,
        "t_window_count": [2, 5, 1, 0, 3],
    })
    deltas = window_deltas(complete, ["g"], "t")
    assert deltas["t_window_delta"].to_list() == [0, 3, -4, 0, 3]
    assert deltas.schema["t_window_delta"] == pl.Int64


def test_window_ranks_are_dense_per_window_and_value():
    complete = pl.DataFrame({
        "g": ["a", "b", "c", "a", "b"],
        "t_window": ["1", "1", "1", "2", "2"],
        "attribute_value": ["v:x"] * 5,
        "t_window_count": [5, 5, 1, 0, 2],
    })
    ranked = window_ranks(complete, "t")
    assert ranked["t_window_rank"].to_list() == [1, 1, 2, 2, 1]


def test_temporal_frame_end_to_end(complaints: pl.DataFrame):
    prepared = prepare_frame(complaints, ["city"])
    frame = temporal_frame(prepared, ["city"], "period", AGGREGATES)
    assert frame.columns == [
        "city",
        "period_window",
        "attribute_value",
        "period_window_count",
        "period_window_rank",
        "period_window_delta",
    ]
    lakeside_price = frame.filter(
        (pl.col("city") == "Lakeside")
        & (pl.col("attribute_value") == "price_issue:True")
    )
    assert rows(
        lakeside_price,
        "period_window",
        "period_window_count",
        "period_window_rank",
        "period_window_delta",
    ) == [("2023-H1", 2, 1, 0), ("2023-H2", 0, 2, -2)]
    keys = rows(frame, "city", "attribute_value", "period_window")
    assert keys == sorted(keys)


def test_temporal_frame_with_no_windows_keeps_the_schema():
    frame = pl.DataFrame({"g": ["a"], "t": [None], "v": ["x"]}).with_columns(
        pl.col("t").cast(pl.Utf8)
    )
    empty = temporal_frame(frame, ["g"], "t", ["v"])
    assert empty.height == 0
    assert empty.schema["t_window_count"] == pl.Int64
    assert empty.schema["t_window_delta"] == pl.Int64
