# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_patterns import PatternConfig
from itk_patterns.models import (
    ATTRIBUTE_TYPE,
    ATTRIBUTE_VALUE,
    FULL_ATTRIBUTE,
    LINK_SCHEMA,
    PERIOD,
    SUBJECT,
)
from itk_patterns.prepare import (
    attribute_columns,
    attribute_links,
    node_labels,
    prepare_frame,
)


def test_prepare_frame_casts_to_text_with_empty_for_missing():
    frame = pl.DataFrame({
        "whole": [1.0, 2.0, None],
        "fraction": [1.5, None, 2.0],
        "flag": [True, False, None],
        "n": [1, 2, 3],
    })
    prepared = prepare_frame(frame)
    assert all(dtype == pl.Utf8 for dtype in prepared.dtypes)
    assert prepared.get_column("whole").to_list() == ["1", "2", ""]
    assert prepared.get_column("fraction").to_list() == ["1.5", "", "2.0"]
    assert prepared.get_column("flag").to_list() == ["True", "False", ""]
    assert prepared.get_column("n").to_list() == ["1", "2", "3"]


def test_attribute_columns_default_excludes_period_and_subject_id(config):
    frame = pl.DataFrame({
        "Subject ID": ["1"],
        "period": ["P1"],
        "a": ["x"],
        "b": ["y"],
    })
    assert attribute_columns(frame, config) == ["a", "b"]


def test_attribute_columns_explicit_keeps_frame_order():
    frame = pl.DataFrame({"period": ["P1"], "a": ["x"], "b": ["y"], "c": ["z"]})
    config = PatternConfig(period_column="period", attribute_columns=("c", "a"))
    assert attribute_columns(frame, config) == ["a", "c"]


def test_attribute_columns_rejects_unknown_or_period(config):
    frame = pl.DataFrame({"period": ["P1"], "a": ["x"]})
    with pytest.raises(ValueError, match="no period column"):
        attribute_columns(frame, PatternConfig(period_column="when"))
    with pytest.raises(ValueError, match="no attribute columns"):
        attribute_columns(
            frame, PatternConfig(period_column="period", attribute_columns=("zzz",))
        )
    with pytest.raises(ValueError, match="cannot also be an attribute"):
        attribute_columns(
            frame, PatternConfig(period_column="period", attribute_columns=("period",))
        )


def test_attribute_links_schema_and_content(cases, config):
    links = attribute_links(prepare_frame(cases), config)
    assert links.schema == LINK_SCHEMA
    # 13 cases x 3 attributes, nothing missing
    assert links.height == 39
    first = links.filter(pl.col(SUBJECT) == 1).sort(ATTRIBUTE_TYPE)
    assert first.get_column(PERIOD).to_list() == ["P1"] * 3
    assert first.get_column(FULL_ATTRIBUTE).to_list() == [
        "city=A",
        "issue=low",
        "product=X",
    ]
    assert first.get_column(ATTRIBUTE_VALUE).to_list() == ["A", "low", "X"]


def test_attribute_links_drops_empty_values_and_empty_periods(config):
    frame = prepare_frame(
        pl.DataFrame({
            "period": ["P1", "", None, "P2"],
            "a": ["x", "x", "x", None],
            "b": ["", "y", "y", "y"],
        })
    )
    links = attribute_links(frame, config)
    # Row 1 keeps only a=x; rows 2 and 3 have no period; row 4 keeps b=y.
    assert links.select([SUBJECT, FULL_ATTRIBUTE]).rows() == [(1, "a=x"), (4, "b=y")]


def test_attribute_links_uses_configured_separator(cases):
    config = PatternConfig(period_column="period", type_value_separator="::")
    links = attribute_links(prepare_frame(cases), config)
    assert "city::A" in links.get_column(FULL_ATTRIBUTE).to_list()


def test_attribute_links_requires_an_attribute_column():
    frame = pl.DataFrame({"period": ["P1"]})
    with pytest.raises(ValueError, match="No attribute columns"):
        attribute_links(frame, PatternConfig(period_column="period"))


def test_node_labels_code_sorted_types_from_zero(cases, config):
    labels = node_labels(attribute_links(prepare_frame(cases), config))
    assert labels == {
        "city=A": 0,
        "city=B": 0,
        "issue=high": 1,
        "issue=low": 1,
        "product=X": 2,
        "product=Y": 2,
    }
