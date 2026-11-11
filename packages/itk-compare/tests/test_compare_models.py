# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import pytest
from itk_compare import AttributeFilter, CompareConfig, ReportSelection, window_columns
from pydantic import ValidationError


def test_attribute_filter_parses_value_with_colons():
    parsed = AttributeFilter.parse("time:12:30")
    assert (parsed.attribute, parsed.value) == ("time", "12:30")
    assert parsed.key == "time:12:30"


def test_attribute_filter_rejects_text_without_separator():
    with pytest.raises(ValueError, match="no ':'"):
        AttributeFilter.parse("city")


def test_config_is_frozen_and_needs_groups_and_aggregates():
    config = CompareConfig(groups=("city",), aggregates=("issue",))
    with pytest.raises(ValidationError):
        config.groups = ("other",)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CompareConfig(groups=(), aggregates=("issue",))
    with pytest.raises(ValidationError):
        CompareConfig(groups=("city",), aggregates=())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": ("city", "city"), "aggregates": ("issue",)},
        {"groups": ("city",), "aggregates": ("issue", "issue")},
        {"groups": ("city",), "aggregates": ("issue",), "temporal": "city"},
        {"groups": ("city",), "aggregates": ("issue",), "temporal": "issue"},
    ],
)
def test_config_rejects_repeated_or_overlapping_columns(kwargs):
    with pytest.raises(ValidationError):
        CompareConfig(**kwargs)


def test_summary_columns_follow_legacy_order():
    plain = CompareConfig(groups=("city", "age"), aggregates=("issue",))
    assert plain.summary_columns == [
        "city",
        "age",
        "group_count",
        "group_rank",
        "attribute_value",
        "group_attribute_count",
        "group_attribute_rank",
    ]
    windowed = CompareConfig(groups=("city",), aggregates=("issue",), temporal="period")
    assert windowed.summary_columns[-4:] == list(window_columns("period"))
    assert window_columns("period").delta == "period_window_delta"


def test_required_columns_include_filter_attributes_once():
    config = CompareConfig(
        groups=("city",),
        aggregates=("issue",),
        temporal="period",
        filters=(
            AttributeFilter(attribute="product", value="A"),
            AttributeFilter(attribute="city", value="Lakeside"),
        ),
    )
    assert config.required_columns() == ["city", "issue", "period", "product"]


def test_report_selection_defaults_to_everything():
    assert ReportSelection().is_empty
    assert not ReportSelection(top_group_ranks=3).is_empty
    with pytest.raises(ValidationError):
        ReportSelection(top_group_ranks=0)
