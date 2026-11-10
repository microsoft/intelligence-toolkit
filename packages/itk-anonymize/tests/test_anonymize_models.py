# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
import pytest
from itk_anonymize.models import AnonymizeConfig, AttributeValue, RawTable


def test_config_defaults_match_legacy_signature() -> None:
    config = AnonymizeConfig(epsilon=12.0)
    assert config.reporting_length == 4
    assert config.percentile_percentage == 99
    assert config.percentile_epsilon_proportion == 0.01
    assert config.number_of_records_epsilon_proportion == 0.005
    assert config.weight_selection_percentile == 95
    assert config.accuracy_mode == "prioritize_long_combinations"
    assert config.fabrication_mode == "balanced"
    assert config.empty_value == ""
    assert config.use_synthetic_counts is True
    assert config.aggregate_counts_scale_factor == 1.0
    assert config.sensitive_zeros == ()
    assert (config.attribute_separator, config.value_separator) == (";", ":")


def test_percentiles_are_integers_as_pacsynth_requires() -> None:
    config = AnonymizeConfig(
        epsilon=1, percentile_percentage=90.0, weight_selection_percentile=80
    )  # type: ignore[arg-type]
    assert isinstance(config.percentile_percentage, int)
    assert isinstance(config.weight_selection_percentile, int)
    with pytest.raises(ValueError, match="percentile_percentage"):
        AnonymizeConfig(epsilon=1, percentile_percentage=99.5)  # type: ignore[arg-type]


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        AnonymizeConfig(epsilon=0)
    with pytest.raises(ValueError, match="reporting_length"):
        AnonymizeConfig(epsilon=1, reporting_length=0)
    with pytest.raises(ValueError, match="fabrication_mode"):
        AnonymizeConfig(epsilon=1, fabrication_mode="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="separator"):
        AnonymizeConfig(epsilon=1, attribute_separator=":", value_separator=":")


def test_config_is_frozen() -> None:
    config = AnonymizeConfig(epsilon=1)
    with pytest.raises(ValueError, match="frozen"):
        config.epsilon = 2  # type: ignore[misc]


def test_attribute_value_is_frozen_and_hashable() -> None:
    item = AttributeValue(attribute="a", value="1")
    assert hash(item) == hash(AttributeValue(attribute="a", value="1"))
    with pytest.raises(ValueError, match="frozen"):
        item.value = "2"  # type: ignore[misc]
    with pytest.raises(ValueError, match="attribute"):
        AttributeValue(attribute="", value="1")


def test_raw_table_checks_row_width() -> None:
    with pytest.raises(ValueError, match="Row 1"):
        RawTable(("a", "b"), (("1", "2"), ("3",)))


def test_raw_table_lists_round_trip() -> None:
    lists = [["a", "b"], ["1", "2"], ["", "3"]]
    table = RawTable.from_lists(lists)
    assert table.as_lists() == lists
    assert table.height == 2
    assert table.to_frame().equals(pl.DataFrame({"a": ["1", ""], "b": ["2", "3"]}))


def test_raw_table_needs_header() -> None:
    with pytest.raises(ValueError, match="header"):
        RawTable.from_lists([])


def test_empty_raw_table_frame_keeps_columns() -> None:
    frame = RawTable(("a", "b"), ()).to_frame()
    assert frame.columns == ["a", "b"]
    assert frame.dtypes == [pl.Utf8, pl.Utf8]
    assert frame.is_empty()
