# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
from itk_anonymize.prepare import (
    prepare_frame,
    suppress_boolean_binary,
    suppress_rare_values,
    to_raw_table,
)


def test_every_column_becomes_text_with_empty_for_null() -> None:
    frame = pl.DataFrame({
        "s": ["a", None],
        "i": [1, None],
        "f": [2.0, None],
        "b": [True, None],
    })
    prepared = prepare_frame(frame)
    assert all(dtype == pl.Utf8 for dtype in prepared.dtypes)
    assert prepared.row(0) == ("a", "1", "2", "True")
    assert prepared.row(1) == ("", "", "", "")


def test_whole_number_floats_lose_their_decimal() -> None:
    frame = pl.DataFrame({"f": [1.0, 3.0, None, float("nan")]})
    assert prepare_frame(frame).get_column("f").to_list() == ["1", "3", "", ""]


def test_fractional_floats_stay_as_written() -> None:
    frame = pl.DataFrame({"f": [1.0, 2.5]})
    assert prepare_frame(frame).get_column("f").to_list() == ["1.0", "2.5"]


def test_booleans_use_pandas_spelling() -> None:
    frame = pl.DataFrame({"b": [True, False]})
    assert prepare_frame(frame).get_column("b").to_list() == ["True", "False"]


def test_prepare_does_not_mutate_input() -> None:
    frame = pl.DataFrame({"f": [1.0]})
    prepare_frame(frame)
    assert frame.dtypes == [pl.Float64]


def test_suppress_boolean_binary_blanks_presence_only_columns() -> None:
    frame = prepare_frame(
        pl.DataFrame({
            "flag": [True, False, None],
            "binary": [0, 1, 0],
            "code": ["0", "1", "2"],
            "name": ["x", "y", "0"],
        })
    )
    out = suppress_boolean_binary(frame)
    assert out.get_column("flag").to_list() == ["True", "", ""]
    assert out.get_column("binary").to_list() == ["", "1", ""]
    # three distinct values: not a presence column
    assert out.get_column("code").to_list() == ["0", "1", "2"]
    # two distinct values but neither is 0/False
    assert out.get_column("name").to_list() == ["x", "y", "0"]


def test_suppress_rare_values_blanks_below_min_count() -> None:
    frame = pl.DataFrame({"a": ["x", "x", "y", ""], "b": ["p", "q", "r", "s"]})
    out = suppress_rare_values(frame, 2)
    assert out.get_column("a").to_list() == ["x", "x", "", ""]
    assert out.get_column("b").to_list() == ["", "", "", ""]


def test_suppress_rare_values_can_target_columns() -> None:
    frame = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    out = suppress_rare_values(frame, 2, columns=["a"])
    assert out.get_column("a").to_list() == ["", ""]
    assert out.get_column("b").to_list() == ["p", "q"]


def test_suppress_rare_values_noop_at_or_below_one() -> None:
    frame = pl.DataFrame({"a": ["x", "y"]})
    assert suppress_rare_values(frame, 0).equals(frame)
    assert suppress_rare_values(frame, 1).equals(frame)


def test_to_raw_table_round_trips() -> None:
    frame = prepare_frame(pl.DataFrame({"a": ["x", None], "b": [1, 2]}))
    table = to_raw_table(frame)
    assert table.header == ("a", "b")
    assert table.rows == (("x", "1"), ("", "2"))
    assert table.to_frame().equals(frame)
