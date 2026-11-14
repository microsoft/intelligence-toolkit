# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import pytest
from itk_anonymize.aggregate import (
    aggregates_frame,
    count_combinations,
    normalize_key,
    parse_aggregates,
    record_attributes,
    render_key,
)
from itk_anonymize.models import RawTable

TABLE = RawTable(
    ("Z", "A", "M"),
    (("z1", "a1", "0"), ("z2", "", "m1"), ("z1", "a1", "m1")),
)

# What pacsynth.Dataset(TABLE).get_aggregates(3, ";") returns for this table.
PACSYNTH_KEYS = {
    "A:a1;M:m1": 1,
    "Z:z1": 2,
    "M:m1;Z:z2": 1,
    "A:a1;Z:z1": 2,
    "Z:z2": 1,
    "A:a1": 2,
    "A:a1;M:m1;Z:z1": 1,
    "M:m1;Z:z1": 1,
    "M:m1": 2,
}


def test_counts_match_pacsynth_including_zero_handling() -> None:
    counted = count_combinations(TABLE, 3)
    assert counted == parse_aggregates(PACSYNTH_KEYS)


def test_sensitive_zeros_make_zero_an_attribute() -> None:
    counted = count_combinations(TABLE, 3, sensitive_zeros=["M"])
    assert counted["M:0",] == 1
    assert counted["A:a1", "M:0", "Z:z1"] == 1


def test_reporting_length_caps_combination_size() -> None:
    counted = count_combinations(TABLE, 1)
    assert all(len(key) == 1 for key in counted)
    assert counted["Z:z1",] == 2


def test_reporting_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="reporting_length"):
        count_combinations(TABLE, 0)


def test_record_attributes_skip_empty_and_zero() -> None:
    assert record_attributes(("A", "B", "C"), ("x", "", "0")) == ["A:x"]
    assert record_attributes(
        ("A", "B", "C"), ("x", "", "0"), sensitive_zeros=["C"]
    ) == [
        "A:x",
        "C:0",
    ]


def test_keys_are_order_insensitive() -> None:
    assert normalize_key(["Z:z", "A:a"]) == ("A:a", "Z:z")
    assert parse_aggregates({"Z:z;A:a": 1, "A:a;Z:z": 2}) == {("A:a", "Z:z"): 3}
    assert render_key(("A:a", "Z:z")) == "A:a;Z:z"


def test_custom_separators() -> None:
    table = RawTable(("A", "B"), (("x", "y"),))
    counted = count_combinations(table, 2, value_separator="=")
    assert ("A=x", "B=y") in counted
    assert parse_aggregates({"A=x|B=y": 1}, "|") == {("A=x", "B=y"): 1}


def test_aggregates_frame_has_record_count_and_stable_order() -> None:
    frame = aggregates_frame({("B:b",): 2, ("A:a",): 2, ("C:c",): 5}, 10)
    assert frame.columns == ["selections", "protected_count"]
    assert frame.get_column("selections").to_list() == [
        "record_count",
        "C:c",
        "A:a",
        "B:b",
    ]
    assert frame.get_column("protected_count").to_list() == [10, 5, 2, 2]


def test_aggregates_frame_of_nothing_still_has_record_count() -> None:
    frame = aggregates_frame({}, 0)
    assert frame.rows() == [("record_count", 0)]
