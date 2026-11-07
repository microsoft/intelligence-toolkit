# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from itk_extract.merge import merge

from .conftest import extraction


def test_empty_input() -> None:
    assert merge([]) == ({}, [])


def test_arrays_concatenate_with_provenance() -> None:
    merged, conflicts = merge([
        extraction("s1#0", {"people": [{"name": "A"}]}),
        extraction("s2#0", {"people": [{"name": "B"}, {"name": "C"}]}),
    ])
    assert conflicts == []
    assert merged["people"] == [
        {"name": "A", "source_id": "s1", "chunk_id": "s1#0"},
        {"name": "B", "source_id": "s2", "chunk_id": "s2#0"},
        {"name": "C", "source_id": "s2", "chunk_id": "s2#0"},
    ]


def test_provenance_can_be_switched_off_and_scalars_in_arrays_untouched() -> None:
    merged, _ = merge(
        [extraction("a#0", {"tags": ["x"]}), extraction("b#0", {"tags": ["y"]})],
        annotate_provenance=False,
    )
    assert merged["tags"] == ["x", "y"]


def test_agreeing_scalars_do_not_conflict_and_empties_are_ignored() -> None:
    merged, conflicts = merge([
        extraction("a#0", {"org": "Acme", "founded": None}),
        extraction("b#0", {"org": "", "founded": 1999}),
        extraction("c#0", {"org": "Acme"}),
    ])
    assert merged == {"org": "Acme", "founded": 1999}
    assert conflicts == []


def test_all_empty_keeps_last_value() -> None:
    merged, _ = merge([extraction("a#0", {"x": None}), extraction("b#0", {"x": ""})])
    assert merged == {"x": ""}


def test_last_policy_reports_conflict() -> None:
    merged, conflicts = merge([
        extraction("a#0", {"org": "Acme"}),
        extraction("b#0", {"org": "ACME Ltd"}),
    ])
    assert merged["org"] == "ACME Ltd"
    assert len(conflicts) == 1
    assert conflicts[0].path == "org"
    assert conflicts[0].chosen == "ACME Ltd"
    assert conflicts[0].values == (("a#0", "Acme"), ("b#0", "ACME Ltd"))


def test_first_policy() -> None:
    merged, _ = merge(
        [extraction("a#0", {"org": "Acme"}), extraction("b#0", {"org": "Other"})],
        policy="first",
    )
    assert merged["org"] == "Acme"


def test_majority_policy_with_tie_goes_to_latest() -> None:
    items = [
        extraction("a#0", {"org": "X"}),
        extraction("b#0", {"org": "Y"}),
        extraction("c#0", {"org": "X"}),
    ]
    assert merge(items, policy="majority")[0]["org"] == "X"
    tie = [extraction("a#0", {"org": "X"}), extraction("b#0", {"org": "Y"})]
    assert merge(tie, policy="majority")[0]["org"] == "Y"


def test_keep_all_policy_lists_distinct_values() -> None:
    merged, conflicts = merge(
        [
            extraction("a#0", {"org": "X"}),
            extraction("b#0", {"org": "Y"}),
            extraction("c#0", {"org": "X"}),
        ],
        policy="keep_all",
    )
    assert merged["org"] == ["X", "Y"]
    assert conflicts[0].chosen == ["X", "Y"]


def test_nested_objects_merge_and_report_dotted_paths() -> None:
    merged, conflicts = merge([
        extraction("a#0", {"hq": {"city": "Oslo", "country": "NO"}}),
        extraction("b#0", {"hq": {"city": "Bergen"}}),
    ])
    assert merged == {"hq": {"city": "Bergen", "country": "NO"}}
    assert [c.path for c in conflicts] == ["hq.city"]


def test_mixed_types_are_treated_as_scalar_conflict() -> None:
    merged, conflicts = merge([
        extraction("a#0", {"x": {"a": 1}}),
        extraction("b#0", {"x": [1]}),
    ])
    assert merged["x"] == [1]
    assert conflicts[0].path == "x"


def test_key_order_follows_first_appearance() -> None:
    merged, _ = merge([
        extraction("a#0", {"b": 1, "a": 2}),
        extraction("c#0", {"c": 3}),
    ])
    assert list(merged) == ["b", "a", "c"]
