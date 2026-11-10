# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
import pytest
from itk_compare import AttributeFilter, CompareConfig
from itk_compare.prepare import (
    apply_filters,
    check_columns,
    filter_options,
    prepare_frame,
)


def test_prepare_casts_to_text_reads_empty_as_null_and_drops_groupless_rows(
    complaints: pl.DataFrame,
):
    prepared = prepare_frame(complaints, ["city"])
    assert prepared.height == 7
    assert all(dtype == pl.Utf8 for dtype in prepared.dtypes)
    assert prepared["price_issue"].null_count() == 1
    assert prepared["city"].null_count() == 0


def test_prepare_casts_numbers_and_booleans_to_text():
    frame = pl.DataFrame({"g": [1, 2], "flag": [True, None], "n": [1.5, 2.0]})
    prepared = prepare_frame(frame, ["g"])
    assert prepared["g"].to_list() == ["1", "2"]
    assert prepared["flag"].to_list() == ["true", None]
    assert prepared["n"].to_list() == ["1.5", "2.0"]


def test_prepare_drops_rows_missing_any_group_column():
    frame = pl.DataFrame({"a": ["x", "x", None], "b": ["y", "", "y"], "v": [1, 2, 3]})
    assert prepare_frame(frame, ["a", "b"])["v"].to_list() == ["1"]


def test_apply_filters_is_an_and_over_text_equality(complaints: pl.DataFrame):
    prepared = prepare_frame(complaints, ["city"])
    filtered = apply_filters(
        prepared,
        [
            AttributeFilter(attribute="product", value="A"),
            AttributeFilter(attribute="price_issue", value="True"),
        ],
    )
    assert filtered["city"].to_list() == ["Lakeside", "Hilltop", "Hilltop"]
    assert apply_filters(prepared, []).height == prepared.height


def test_filter_options_are_sorted_and_skip_missing_markers():
    frame = pl.DataFrame({
        "b": ["2", "1", None, ""],
        "a": ["x", "nan", "None", "x"],
    })
    assert filter_options(frame) == ["a:x", "b:1", "b:2"]


def test_check_columns_names_every_missing_column(complaints: pl.DataFrame):
    config = CompareConfig(
        groups=("city",),
        aggregates=("missing_a",),
        temporal="missing_t",
        filters=(AttributeFilter(attribute="product", value="A"),),
    )
    with pytest.raises(KeyError, match=r"\['missing_a', 'missing_t'\]"):
        check_columns(complaints, config)
