# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
import pytest
from itk_anonymize.models import AttributeValue
from itk_anonymize.selection import (
    data_schema,
    describe_selection,
    selection_from_dicts,
    selection_keys,
)
from itk_anonymize_testing import av


def test_attribute_value_key_and_parse_round_trip() -> None:
    item = av("city", "Lakeside")
    assert item.key() == "city:Lakeside"
    assert AttributeValue.parse("city:Lakeside") == item
    assert AttributeValue.parse("city=Lake:side", "=") == av("city", "Lake:side")


def test_parse_requires_separator() -> None:
    with pytest.raises(ValueError, match="no ':'"):
        AttributeValue.parse("city")


def test_selection_from_app_dicts() -> None:
    selection = selection_from_dicts([{"attribute": "a", "value": "1"}])
    assert selection == [av("a", "1")]
    assert selection_keys(selection) == ["a:1"]


def test_describe_multiline_sorts_values_within_attribute() -> None:
    text = describe_selection([av("city", "B"), av("age", "30"), av("city", "A")])
    assert text == "- city = A | B\n- age = 30\n"


def test_describe_single_line() -> None:
    text = describe_selection(
        [av("city", "B"), av("city", "A"), av("age", "30")], False
    )
    assert text == "city:A|B, age:30"


def test_describe_empty() -> None:
    assert describe_selection([]) == ""
    assert describe_selection([], multiline=False) == ""


def test_data_schema_sorted_without_empties() -> None:
    frame = pl.DataFrame({"Color": ["Red", "Blue", "", "Red"], "n": [2, 1, None, 2]})
    assert data_schema(frame) == {"Color": ["Blue", "Red"], "n": ["1", "2"]}


def test_data_schema_of_none_or_empty() -> None:
    assert data_schema(None) == {}
    assert data_schema(pl.DataFrame({"a": []})) == {"a": []}
