# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from itk_mock import TextConfig
from itk_mock.frames import extract_frame, flatten, records_at, to_frames
from itk_mock.texts import generate_texts, record_to_input_text, records_to_input_texts

if TYPE_CHECKING:
    from itk_mock_testing import FakeChat

OBJECT = {
    "dataset": "demo",
    "records": [
        {
            "name": "Ana",
            "age": 31,
            "address": {"city": "Lakeside", "geo": {"lat": 1.5}},
            "tags": ["a", "b"],
            "orders": [{"sku": "x1"}, {"sku": "x2"}],
        },
        {
            "name": "Bo",
            "age": 40.5,
            "address": {"city": "Hilltop", "geo": {"lat": 2.0}},
            "tags": [],
            "orders": [{"sku": "y1"}],
            "nickname": "B",
        },
    ],
}


def test_records_at_fans_out_through_arrays_and_objects() -> None:
    assert [r["name"] for r in records_at(OBJECT, ["records"])] == ["Ana", "Bo"]
    assert records_at(OBJECT, ["records", "orders"]) == [
        {"sku": "x1"},
        {"sku": "x2"},
        {"sku": "y1"},
    ]
    assert records_at(OBJECT, ["records", "tags"]) == ["a", "b"]
    assert records_at({"wrapper": {"rows": [1, 2]}}, ["wrapper", "rows"]) == [1, 2]
    assert records_at(OBJECT, ["missing"]) == []
    assert records_at(OBJECT, ["dataset"]) == []


def test_flatten_dots_nested_objects_and_keeps_lists() -> None:
    assert flatten(OBJECT["records"][0]) == {
        "name": "Ana",
        "age": 31,
        "tags": ["a", "b"],
        "orders": [{"sku": "x1"}, {"sku": "x2"}],
        "address.city": "Lakeside",
        "address.geo.lat": 1.5,
    }
    assert list(flatten(OBJECT["records"][0])) == [
        "name",
        "age",
        "tags",
        "orders",
        "address.city",
        "address.geo.lat",
    ]
    assert flatten("a") == {"0": "a"}


def test_extract_frame_unions_columns_in_first_seen_order() -> None:
    frame = extract_frame(OBJECT, ["records"])

    assert frame.columns == [
        "name",
        "age",
        "tags",
        "orders",
        "address.city",
        "address.geo.lat",
        "nickname",
    ]
    assert frame.get_column("age").to_list() == [31.0, 40.5]
    assert frame.get_column("nickname").to_list() == [None, "B"]
    assert frame.get_column("tags").to_list() == [["a", "b"], []]


def test_to_frames_keys_by_dotted_path_and_handles_empty_arrays() -> None:
    frames = to_frames(
        OBJECT, [["records"], ["records", "tags"], ["records", "orders"], ["nothing"]]
    )

    assert list(frames) == ["records", "records.tags", "records.orders", "nothing"]
    assert frames["records.orders"].get_column("sku").to_list() == ["x1", "x2", "y1"]
    assert frames["records.tags"].get_column("0").to_list() == ["a", "b"]
    assert frames["nothing"].shape == (0, 0)


def test_record_to_input_text_is_compact_json() -> None:
    record = {"name": "Ana", "age": 31, "ok": True, "note": None, "tags": ["a"]}

    assert (
        record_to_input_text(record)
        == '{"name":"Ana","age":31,"ok":true,"note":null,"tags":["a"]}'
    )


def test_records_to_input_texts_accepts_frames_and_dicts() -> None:
    rows = [{"name": "Ana", "age": 31}, {"name": "Bo", "age": 40}]

    expected = ['{"name":"Ana","age":31}', '{"name":"Bo","age":40}']
    assert records_to_input_texts(rows) == expected
    assert records_to_input_texts(pl.DataFrame(rows)) == expected


async def test_generate_texts_keeps_input_order_and_reports_each_batch(
    fake_chat: FakeChat,
) -> None:
    inputs = [f'{{"id":{i}}}' for i in range(5)]
    config = TextConfig(
        generation_guidance="Formal tone", temperature=0.9, parallelism=2
    )
    updates: list[int] = []

    result = await generate_texts(
        fake_chat, inputs, config, on_update=lambda frame: updates.append(frame.height)
    )

    assert result.texts == [f"Text about {text}" for text in inputs]
    assert result.input_texts == inputs
    assert result.frame.columns == ["mock_text"]
    assert result.frame.get_column("mock_text").to_list() == result.texts
    assert updates == [2, 4, 5]
    assert len(fake_chat.calls) == 5
    assert all(call["response_schema"] is None for call in fake_chat.calls)
    assert all(call["temperature"] == 0.9 for call in fake_chat.calls)
    assert all("Formal tone" in prompt for prompt in fake_chat.prompts)


async def test_generate_texts_with_no_inputs_gives_empty_text_frame(
    fake_chat: FakeChat,
) -> None:
    result = await generate_texts(fake_chat, [])

    assert result.texts == []
    assert result.frame.schema == {"mock_text": pl.Utf8}
    assert fake_chat.calls == []
