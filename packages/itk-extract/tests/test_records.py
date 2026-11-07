# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from typing import TYPE_CHECKING

from itk_extract.records import flatten, records_at, records_frame, to_frames

if TYPE_CHECKING:
    from itk_extract.schema import RecordSchema

DATA = {
    "organisation": "Acme",
    "people": [
        {
            "name": "Alice",
            "contact": {"email": "a@x"},
            "events": [{"date": "2024-01-02", "description": "A1"}],
            "source_id": "s1",
            "chunk_id": "s1#0",
        },
        {
            "name": "Bob",
            "contact": {"email": "b@x"},
            "events": [
                {"date": "2023-05-06", "description": "B1"},
                {"date": "2024-02-03", "description": "B2"},
            ],
            "source_id": "s2",
            "chunk_id": "s2#0",
        },
    ],
}


def test_records_at_top_level_and_nested() -> None:
    assert [p["name"] for p in records_at(DATA, ("people",))] == ["Alice", "Bob"]
    nested = records_at(DATA, ("people", "events"))
    assert [e["description"] for e in nested] == ["A1", "B1", "B2"]


def test_records_at_missing_or_wrong_shape() -> None:
    assert records_at(DATA, ("nope",)) == []
    assert records_at(DATA, ("organisation",)) == []
    assert records_at({"people": None}, ("people", "events")) == []


def test_flatten_dotted_keys_keeps_lists_and_wraps_scalars() -> None:
    row = flatten({"a": {"b": {"c": 1}}, "tags": ["x"], "n": 2})
    assert row == {"a.b.c": 1, "tags": ["x"], "n": 2}
    assert flatten("plain") == {"value": "plain"}


def test_records_frame_puts_provenance_last() -> None:
    frame = records_frame(DATA, ("people",))
    assert frame.columns[-2:] == ["source_id", "chunk_id"]
    assert frame.columns[:2] == ["name", "contact.email"]
    assert frame.height == 2


def test_records_frame_empty() -> None:
    assert records_frame(DATA, ("nope",)).is_empty()


def test_records_frame_dedupe_ignores_provenance() -> None:
    data = {
        "people": [
            {"name": "A", "source_id": "s1", "chunk_id": "s1#0"},
            {"name": "A", "source_id": "s2", "chunk_id": "s2#0"},
            {"name": "B", "source_id": "s2", "chunk_id": "s2#0"},
        ]
    }
    frame = records_frame(data, ("people",), dedupe=True)
    assert frame.get_column("name").to_list() == ["A", "B"]
    assert frame.get_column("source_id").to_list() == ["s1", "s2"]


def test_to_frames_one_per_array_path(schema: RecordSchema) -> None:
    frames = to_frames(DATA, schema)
    assert set(frames) == {"people", "people.events"}
    assert frames["people.events"].height == 3
    assert frames["people.events"].get_column("description").to_list() == [
        "A1",
        "B1",
        "B2",
    ]
