# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
from itk_compare.groups import (
    attribute_counts,
    count_attribute_values,
    group_counts,
)
from itk_compare.prepare import prepare_frame
from itk_compare_testing import rows


def test_group_counts_rank_with_shared_lower_rank_on_ties(complaints: pl.DataFrame):
    prepared = prepare_frame(complaints, ["city"])
    counts = group_counts(prepared, ["city"])
    assert rows(counts, "city", "group_count", "group_rank") == [
        ("Hilltop", 3, 2),
        ("Lakeside", 3, 2),
        ("Seaside", 1, 3),
    ]
    assert counts.schema["group_count"] == pl.Int64
    assert counts.schema["group_rank"] == pl.Int64


def test_group_counts_over_two_columns_sorts_by_both():
    frame = pl.DataFrame({"a": ["y", "x", "x", "y"], "b": ["1", "2", "1", "1"]})
    counts = group_counts(frame, ["a", "b"])
    assert rows(counts, "a", "b", "group_count", "group_rank") == [
        ("x", "1", 1, 3),
        ("x", "2", 1, 3),
        ("y", "1", 2, 1),
    ]


def test_count_attribute_values_skips_missing_values(complaints: pl.DataFrame):
    prepared = prepare_frame(complaints, ["city"])
    counted = count_attribute_values(
        prepared, ["city"], ["price_issue", "quality_issue"], "n"
    )
    assert counted.columns == ["city", "attribute_value", "n"]
    assert counted.schema["n"] == pl.Int64
    # 7 records x 2 aggregates, minus the two blank cells.
    assert counted["n"].sum() == 12
    assert (
        counted.filter(pl.col("attribute_value") == "price_issue:True")["n"].sum() == 4
    )


def test_count_attribute_values_merges_columns_rendering_the_same_text():
    frame = pl.DataFrame({"g": ["x", "x"], "a": ["b:c", None], "a:b": [None, "c"]})
    counted = count_attribute_values(frame, ["g"], ["a", "a:b"], "n")
    assert rows(counted, "g", "attribute_value", "n") == [("x", "a:b:c", 2)]


def test_count_attribute_values_without_aggregates_is_empty():
    frame = pl.DataFrame({"g": ["x"]})
    counted = count_attribute_values(frame, ["g"], [], "n")
    assert counted.height == 0
    assert counted.columns == ["g", "attribute_value", "n"]


def test_attribute_counts_cross_every_group_with_every_value(complaints: pl.DataFrame):
    prepared = prepare_frame(complaints, ["city"])
    counts = attribute_counts(prepared, ["city"], ["price_issue", "quality_issue"])
    assert counts.height == 3 * 4
    by_value = {
        (city, value): (count, rank)
        for city, value, count, rank in rows(
            counts,
            "city",
            "attribute_value",
            "group_attribute_count",
            "group_attribute_rank",
        )
    }
    assert by_value["Lakeside", "price_issue:True"] == (2, 2)
    assert by_value["Hilltop", "price_issue:True"] == (2, 2)
    assert by_value["Seaside", "price_issue:True"] == (0, 3)
    assert by_value["Hilltop", "quality_issue:True"] == (3, 1)
    assert by_value["Seaside", "quality_issue:True"] == (0, 3)
    assert by_value["Hilltop", "price_issue:False"] == (0, 3)
    assert by_value["Seaside", "price_issue:False"] == (1, 2)


def test_attribute_counts_are_sorted_by_group_then_value(complaints: pl.DataFrame):
    prepared = prepare_frame(complaints, ["city"])
    counts = attribute_counts(prepared, ["city"], ["price_issue", "quality_issue"])
    keys = rows(counts, "city", "attribute_value")
    assert keys == sorted(keys)
    assert counts.schema["group_attribute_count"] == pl.Int64
    assert counts.schema["group_attribute_rank"] == pl.Int64


def test_attribute_counts_include_groups_with_no_values_at_all():
    frame = pl.DataFrame({"g": ["a", "b"], "v": ["x", None]})
    counts = attribute_counts(frame, ["g"], ["v"])
    assert rows(counts, "g", "attribute_value", "group_attribute_count") == [
        ("a", "v:x", 1),
        ("b", "v:x", 0),
    ]


def test_attribute_counts_of_an_empty_frame_are_empty():
    frame = pl.DataFrame(schema={"g": pl.Utf8, "v": pl.Utf8})
    counts = attribute_counts(frame, ["g"], ["v"])
    assert counts.height == 0
    assert counts.columns == [
        "g",
        "attribute_value",
        "group_attribute_count",
        "group_attribute_rank",
    ]
