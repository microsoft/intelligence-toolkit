# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
import pytest
from itk_anonymize.aggregate import parse_aggregates
from itk_anonymize.queries import (
    aggregate_links,
    synthetic_links,
    time_series,
    top_attributes,
)
from itk_anonymize_testing import av

SYNTHETIC = pl.DataFrame({
    "Color": ["Red", "Blue", "Red", "Green"],
    "Size": ["Large", "Small", "Large", ""],
})
COLUMNS = ["Attribute", "Attribute Value", "Count", "Dataset"]
LINK_COLUMNS = ["Source", "Target", "Count", "Highlight", "Proportion", "Dataset"]


def rows(frame: pl.DataFrame) -> set[tuple]:
    return set(frame.rows())


# ---- top_attributes ---------------------------------------------------------


def test_top_attributes_synthetic_only_when_no_aggregates() -> None:
    result = top_attributes(SYNTHETIC, {})
    assert result.columns == COLUMNS
    assert rows(result) == {
        ("Color", "Color:Red", 2, "Synthetic"),
        ("Size", "Size:Large", 2, "Synthetic"),
        ("Color", "Color:Blue", 1, "Synthetic"),
        ("Color", "Color:Green", 1, "Synthetic"),
        ("Size", "Size:Small", 1, "Synthetic"),
    }


def test_top_attributes_prefers_exact_aggregate_counts() -> None:
    aggregates = parse_aggregates({"Color:Red": 15, "Color:Blue": 8, "Size:Large": 12})
    result = top_attributes(SYNTHETIC, aggregates)
    by_value = {r[1]: (r[2], r[3]) for r in result.rows()}
    assert by_value["Color:Red"] == (15, "Aggregate")
    assert by_value["Color:Blue"] == (8, "Aggregate")
    assert by_value["Size:Large"] == (12, "Aggregate")
    # not in the aggregates: synthetic count kept
    assert by_value["Color:Green"] == (1, "Synthetic")
    assert by_value["Size:Small"] == (1, "Synthetic")


def test_top_attributes_filters_rows_and_looks_up_extended_combination() -> None:
    aggregates = parse_aggregates({
        "Color:Red": 15,
        "Color:Red;Size:Large": 11,
        "Size:Large": 20,
    })
    result = top_attributes(SYNTHETIC, aggregates, [av("Color", "Red")])
    by_value = {r[1]: (r[2], r[3]) for r in result.rows()}
    # the selected value itself reports the count of the selection
    assert by_value["Color:Red"] == (15, "Aggregate")
    # other values report the count of selection + value, not the bare value
    assert by_value["Size:Large"] == (11, "Aggregate")
    assert set(by_value) == {"Color:Red", "Size:Large"}


def test_top_attributes_union_selection_uses_synthetic_counts_only() -> None:
    aggregates = parse_aggregates({"Color:Red": 15, "Color:Blue": 8, "Size:Large": 12})
    result = top_attributes(
        SYNTHETIC, aggregates, [av("Color", "Red"), av("Color", "Blue")]
    )
    assert result.get_column("Dataset").unique().to_list() == ["Synthetic"]
    assert rows(result) == {
        ("Color", "Color:Red", 2, "Synthetic"),
        ("Color", "Color:Blue", 1, "Synthetic"),
        ("Size", "Size:Large", 2, "Synthetic"),
        ("Size", "Size:Small", 1, "Synthetic"),
    }


def test_top_attributes_ignores_selection_values_absent_from_synthetic() -> None:
    # legacy behaviour: a value the synthetic data never shows filters nothing
    result = top_attributes(SYNTHETIC, {}, [av("Color", "Purple")])
    assert result.height == 5


def test_top_attributes_show_attributes_and_num_values() -> None:
    result = top_attributes(SYNTHETIC, {}, show_attributes=["Color"])
    assert result.get_column("Attribute").unique().to_list() == ["Color"]
    assert top_attributes(SYNTHETIC, {}, num_values=2).height == 2
    assert top_attributes(SYNTHETIC, {}, num_values=0).height == 5


def test_top_attributes_sorted_by_count_then_value() -> None:
    result = top_attributes(SYNTHETIC, {})
    assert result.get_column("Attribute Value").to_list() == [
        "Color:Red",
        "Size:Large",
        "Color:Blue",
        "Color:Green",
        "Size:Small",
    ]


def test_top_attributes_custom_value_separator() -> None:
    aggregates = parse_aggregates({"Color=Red": 7})
    result = top_attributes(SYNTHETIC, aggregates, value_separator="=")
    by_value = {r[1]: (r[2], r[3]) for r in result.rows()}
    assert by_value["Color=Red"] == (7, "Aggregate")
    assert by_value["Size=Small"] == (1, "Synthetic")


def test_top_attributes_empty_synthetic() -> None:
    empty = pl.DataFrame(
        {"Color": [], "Size": []}, schema={"Color": pl.Utf8, "Size": pl.Utf8}
    )
    result = top_attributes(empty, {})
    assert result.columns == COLUMNS
    assert result.is_empty()


# ---- time_series ------------------------------------------------------------


def test_time_series_one_block_per_time_with_zero_fill() -> None:
    synthetic = pl.DataFrame({
        "Color": ["Red", "Blue", "Red", "Red"],
        "Time": ["2023", "2023", "2024", ""],
    })
    result = time_series(synthetic, {}, [], "Time", ["Color"])
    assert result.columns == ["Time", *COLUMNS]
    assert rows(result) == {
        ("2023", "Color", "Color:Red", 1, "Synthetic"),
        ("2023", "Color", "Color:Blue", 1, "Synthetic"),
        ("2024", "Color", "Color:Red", 1, "Synthetic"),
        ("2024", "Color", "Color:Blue", 0, "Aggregate"),
    }
    assert result.get_column("Time").to_list() == ["2023", "2023", "2024", "2024"]


def test_time_series_uses_aggregates_per_time() -> None:
    synthetic = pl.DataFrame({"Color": ["Red", "Blue"], "Time": ["2023", "2024"]})
    aggregates = parse_aggregates({"Color:Red;Time:2023": 9, "Color:Blue;Time:2024": 4})
    result = time_series(synthetic, aggregates, [], "Time", ["Color"])
    assert rows(result) == {
        ("2023", "Color", "Color:Red", 9, "Aggregate"),
        ("2023", "Color", "Color:Blue", 0, "Aggregate"),
        ("2024", "Color", "Color:Blue", 4, "Aggregate"),
        ("2024", "Color", "Color:Red", 0, "Aggregate"),
    }


def test_time_series_without_times_is_empty() -> None:
    synthetic = pl.DataFrame({"Color": ["Red"], "Time": [""]})
    result = time_series(synthetic, {}, [], "Time", ["Color"])
    assert result.columns == ["Time", *COLUMNS]
    assert result.is_empty()


def test_time_series_requires_time_column() -> None:
    with pytest.raises(ValueError, match="Time"):
        time_series(SYNTHETIC, {}, [], "Time", ["Color"])


# ---- aggregate_links --------------------------------------------------------


def test_aggregate_links_basic() -> None:
    aggregates = parse_aggregates({
        "Color:Red;Size:Large": 10,
        "Color:Blue;Size:Small": 5,
        "Color:Red": 20,
    })
    result = aggregate_links(aggregates, [], "Color", "Size")
    assert result.columns == LINK_COLUMNS
    assert rows(result) == {
        ("Red", "Large", 10, 0, 0.0, "Aggregate"),
        ("Blue", "Small", 5, 0, 0.0, "Aggregate"),
    }


def test_aggregate_links_with_filter_and_highlight() -> None:
    aggregates = parse_aggregates({
        "Color:Red;Size:Large": 10,  # lacks the filter
        "Color:Red;Region:North;Size:Large": 6,
        "Color:Red;Region:North;Size:Large;Shade:Dark": 3,
        "Color:Red;Region:North;Size:Large;Shade:Light": 2,
        "Color:Blue;Region:North;Size:Small": 0,  # zero counts dropped
    })
    result = aggregate_links(
        aggregates, [av("Region", "North")], "Color", "Size", av("Shade", "Dark")
    )
    assert rows(result) == {("Red", "Large", 6, 3, 0.5, "Aggregate")}


def test_aggregate_links_ignores_record_count_style_keys() -> None:
    result = aggregate_links({("record_count",): 99}, [], "Color", "Size")
    assert result.is_empty()


def test_aggregate_links_values_may_contain_separator() -> None:
    aggregates = parse_aggregates({"Color:Red;Time:12:30": 1})
    result = aggregate_links(aggregates, [], "Color", "Time")
    assert result.rows() == [("Red", "12:30", 1, 0, 0.0, "Aggregate")]


# ---- synthetic_links --------------------------------------------------------


def test_synthetic_links_basic_skips_empty_values() -> None:
    result = synthetic_links(SYNTHETIC, [], "Color", "Size")
    assert result.columns == LINK_COLUMNS
    assert rows(result) == {
        ("Red", "Large", 2, 0, 0.0, "Synthetic"),
        ("Blue", "Small", 1, 0, 0.0, "Synthetic"),
    }


def test_synthetic_links_with_filters_and_highlight() -> None:
    synthetic = pl.DataFrame({
        "Color": ["Red", "Red", "Red", "Blue"],
        "Size": ["Large", "Large", "Small", "Small"],
        "Region": ["North", "South", "North", "North"],
        "Shade": ["Dark", "Dark", "Light", "Dark"],
    })
    result = synthetic_links(
        synthetic,
        [av("Region", "North"), av("Region", "South")],
        "Color",
        "Size",
        av("Shade", "Dark"),
    )
    assert rows(result) == {
        ("Red", "Large", 2, 2, 1.0, "Synthetic"),
        ("Red", "Small", 1, 0, 0.0, "Synthetic"),
        ("Blue", "Small", 1, 1, 1.0, "Synthetic"),
    }
    assert result.get_column("Source").to_list() == ["Blue", "Red", "Red"]


def test_synthetic_links_requires_columns() -> None:
    with pytest.raises(ValueError, match="Shade"):
        synthetic_links(SYNTHETIC, [], "Color", "Size", av("Shade", "Dark"))
    with pytest.raises(ValueError, match="Region"):
        synthetic_links(SYNTHETIC, [av("Region", "N")], "Color", "Size")
