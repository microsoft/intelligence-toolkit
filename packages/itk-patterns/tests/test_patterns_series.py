# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
from itk_patterns import PatternConfig
from itk_patterns.models import (
    ATTRIBUTE_COUNT_SCHEMA,
    ATTRIBUTE_VALUE,
    COUNT,
    PATTERN,
    PERIOD,
    TIME_SERIES_SCHEMA,
)
from itk_patterns.series import attribute_counts, pattern_filter, time_series


def test_time_series_one_row_per_pattern_and_period(prepared):
    series = time_series(
        prepared.links, ["city=A & product=X", "city=A & product=X", "city=B"]
    )
    assert series.schema == TIME_SERIES_SCHEMA
    assert (
        series.get_column(PATTERN).to_list()
        == ["city=A & product=X"] * 3 + ["city=B"] * 3
    )
    assert series.filter(pl.col(PATTERN) == "city=A & product=X").get_column(
        COUNT
    ).to_list() == [1, 1, 4]
    assert series.get_column(PERIOD).to_list()[:3] == ["P1", "P2", "P3"]


def test_time_series_empty_without_patterns(prepared):
    assert time_series(prepared.links, []).schema == TIME_SERIES_SCHEMA
    assert time_series(prepared.links, []).height == 0


def test_pattern_filter_skips_unknown_columns_and_bad_terms(cases, config):
    expression = pattern_filter(
        "city=A & nope=1 & malformed", "P3", config, cases.columns
    )
    assert cases.filter(expression).height == 4


def test_attribute_counts_for_pattern_in_period(prepared):
    counts = attribute_counts(
        prepared.source, "city=A & product=X", "P3", prepared.config
    )
    assert counts.schema == ATTRIBUTE_COUNT_SCHEMA
    assert counts.rows() == [
        ("city=A", 4),
        ("product=X", 4),
        ("issue=high", 3),
        ("issue=low", 1),
    ]


def test_attribute_counts_ignore_empty_values_and_subject_id():
    source = pl.DataFrame({
        "Subject ID": ["1", "2"],
        "period": ["P1", "P1"],
        "a": ["x", "x"],
        "b": ["", "y"],
    })
    counts = attribute_counts(
        source, "a=x", "P1", PatternConfig(period_column="period")
    )
    assert counts.rows() == [("a=x", 2), ("b=y", 1)]


def test_attribute_counts_empty_when_nothing_matches(prepared):
    counts = attribute_counts(prepared.source, "city=Z", "P3", prepared.config)
    assert counts.height == 0
    assert counts.columns == [ATTRIBUTE_VALUE, COUNT]
