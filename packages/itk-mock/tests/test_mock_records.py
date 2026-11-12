# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import json
import random
from typing import Any

import pytest
from itk_mock import MockConfig, MockDataError
from itk_mock.records import (
    generate_records,
    merge_json_objects,
    merge_objects,
    parse_reply,
    sample_from_record_array,
    select_random_records,
)
from itk_mock_testing import FakeChat


def test_parse_reply_returns_the_object() -> None:
    assert parse_reply('{"records": []}', 0) == {"records": []}


def test_parse_reply_rejects_non_json_and_non_objects() -> None:
    with pytest.raises(MockDataError, match="Batch 2: AI did not return a valid JSON"):
        parse_reply("sorry, no", 2)
    with pytest.raises(MockDataError, match="Batch 1: AI returned a JSON list"):
        parse_reply("[1, 2]", 1)


def test_merge_concatenates_arrays_and_merges_objects() -> None:
    first = {"records": [1], "meta": {"source": "a", "ids": [1]}, "only_first": True}
    second = {"records": [2], "meta": {"source": "a", "ids": [2]}, "only_second": 0}

    merged, conflicts = merge_json_objects(first, second)

    assert merged == {
        "records": [1, 2],
        "meta": {"source": "a", "ids": [1, 2]},
        "only_first": True,
        "only_second": 0,
    }
    assert list(merged) == ["records", "meta", "only_first", "only_second"]
    assert conflicts == []
    assert first["records"] == [1]


def test_merge_takes_second_scalar_and_reports_dotted_conflicts() -> None:
    first = {"title": "a", "meta": {"version": 1, "kind": "x"}, "mixed": [1]}
    second = {"title": "b", "meta": {"version": 2, "kind": "x"}, "mixed": "text"}

    merged, conflicts = merge_json_objects(first, second)

    assert merged == {
        "title": "b",
        "meta": {"version": 2, "kind": "x"},
        "mixed": "text",
    }
    assert conflicts == ["title", "meta.version", "mixed"]


def test_merge_objects_folds_in_order() -> None:
    merged, conflicts = merge_objects([
        {"dataset": "a", "records": [1]},
        {"dataset": "b", "records": [2]},
        {"dataset": "c", "records": [3]},
    ])

    assert merged == {"dataset": "c", "records": [1, 2, 3]}
    assert conflicts == ["dataset", "dataset"]
    assert merge_objects([]) == ({}, [])


def test_sample_returns_everything_when_not_enough_records() -> None:
    obj = {"records": [{"id": 1}, {"id": 2}]}

    assert sample_from_record_array(obj, ["records"], 5, random.Random(0)) == [
        {"id": 1},
        {"id": 2},
    ]


def test_sample_is_reproducible_for_a_seed() -> None:
    obj = {"wrapper": {"records": [{"id": i} for i in range(20)]}}
    path = ["wrapper", "records"]

    first = sample_from_record_array(obj, path, 3, random.Random(7))
    second = sample_from_record_array(obj, path, 3, random.Random(7))

    assert first == second
    assert first == random.Random(7).sample(obj["wrapper"]["records"], 3)
    assert len({r["id"] for r in first}) == 3


def test_select_random_records_gives_disjoint_ids_per_category() -> None:
    counts = {"duplicate": 2, "related": 3}

    first = select_random_records(20, counts, random.Random(1))
    second = select_random_records(20, counts, random.Random(1))

    assert first == second
    assert [len(first["duplicate"]), len(first["related"])] == [2, 3]
    assert set(first["duplicate"]).isdisjoint(first["related"])
    assert all(0 <= i < 20 for ids in first.values() for i in ids)


async def test_single_batch_makes_one_unseeded_call(
    fake_chat: FakeChat, people_schema: dict[str, Any]
) -> None:
    config = MockConfig(
        num_records_overall=3,
        records_per_batch=3,
        temperature=0.2,
        generation_guidance="Only Lakeside residents",
    )

    result = await generate_records(fake_chat, people_schema, config)

    assert len(fake_chat.calls) == 1
    call = fake_chat.calls[0]
    assert call["response_schema"] is people_schema
    assert call["temperature"] == 0.2
    assert "Only Lakeside residents" in fake_chat.prompts[0]
    assert "['records']" in fake_chat.prompts[0]
    assert "Output record count: 3" in fake_chat.prompts[0]
    assert result.json_object["dataset"] == "unseeded"
    assert [r["name"] for r in result.json_object["records"]] == [
        "unseeded/0",
        "unseeded/1",
        "unseeded/2",
    ]
    assert result.generated_objects == [result.json_object]
    assert result.conflicts == []
    assert result.record_arrays == [
        ["records"],
        ["records", "tags"],
        ["records", "orders"],
    ]
    assert result.primary_record_array == ["records"]
    assert result.primary_frame.height == 3


async def test_seeded_batches_use_sampled_records_and_targets(
    fake_chat: FakeChat, people_schema: dict[str, Any]
) -> None:
    config = MockConfig(
        num_records_overall=12,
        records_per_batch=4,
        duplicate_records_per_batch=1,
        related_records_per_batch=2,
        seed=11,
    )

    result = await generate_records(fake_chat, people_schema, config)

    assert len(fake_chat.calls) == 3
    first_batch = result.generated_objects[0]["records"]
    expected_seeds = random.Random(11).sample(first_batch, 2)
    for prompt, seed in zip(fake_chat.prompts[1:], expected_seeds, strict=True):
        assert str(seed) in prompt
        assert "Total records: 4" in prompt
        assert "Near duplicates of seed: 1" in prompt
        assert "Close relations of seed: 2" in prompt
    assert len(result.json_object["records"]) == 12
    assert result.json_object["dataset"] == expected_seeds[1]["name"]
    assert result.conflicts == ["dataset", "dataset"]
    assert result.frames["records"].height == 12
    assert result.frames["records.orders"].height == 12


async def test_same_seed_gives_same_object_and_other_seed_differs(
    people_schema: dict[str, Any],
) -> None:
    def config(seed: int) -> MockConfig:
        return MockConfig(num_records_overall=30, records_per_batch=10, seed=seed)

    first = await generate_records(FakeChat(), people_schema, config(5))
    second = await generate_records(FakeChat(), people_schema, config(5))
    other = await generate_records(FakeChat(), people_schema, config(6))

    assert first.json_object == second.json_object
    assert first.json_object != other.json_object


async def test_explicit_rng_overrides_config_seed(
    people_schema: dict[str, Any],
) -> None:
    config = MockConfig(num_records_overall=30, records_per_batch=10, seed=5)

    seeded = await generate_records(FakeChat(), people_schema, config)
    explicit = await generate_records(
        FakeChat(), people_schema, config, rng=random.Random(5)
    )
    different = await generate_records(
        FakeChat(), people_schema, config, rng=random.Random(99)
    )

    assert explicit.json_object == seeded.json_object
    assert different.json_object != seeded.json_object


async def test_rounds_are_capped_by_parallel_batches(
    fake_chat: FakeChat, people_schema: dict[str, Any]
) -> None:
    config = MockConfig(
        num_records_overall=50, records_per_batch=10, parallel_batches=2, seed=0
    )
    updates: list[int] = []

    result = await generate_records(
        fake_chat,
        people_schema,
        config,
        on_update=lambda frames: updates.append(frames["records"].height),
    )

    assert len(fake_chat.calls) == 5
    assert updates == [10, 30, 50]
    assert len(result.generated_objects) == 5


async def test_keeps_generating_when_seeds_are_scarcer_than_batches(
    fake_chat: FakeChat, people_schema: dict[str, Any]
) -> None:
    config = MockConfig(num_records_overall=20, records_per_batch=2, seed=0)
    updates: list[int] = []

    result = await generate_records(
        fake_chat,
        people_schema,
        config,
        on_update=lambda frames: updates.append(frames["records"].height),
    )

    assert updates == [2, 6, 18, 20]
    assert len(result.json_object["records"]) == 20


async def test_partial_last_batch_still_requests_a_full_batch(
    fake_chat: FakeChat, people_schema: dict[str, Any]
) -> None:
    config = MockConfig(num_records_overall=15, records_per_batch=10, seed=0)

    result = await generate_records(fake_chat, people_schema, config)

    assert len(fake_chat.calls) == 2
    assert len(result.json_object["records"]) == 20


async def test_schema_without_arrays_raises(fake_chat: FakeChat) -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    with pytest.raises(MockDataError, match="no array field"):
        await generate_records(fake_chat, schema)
    assert fake_chat.calls == []


async def test_invalid_first_reply_raises(people_schema: dict[str, Any]) -> None:
    chat = FakeChat(reply=lambda messages, schema: "not json")

    with pytest.raises(MockDataError, match="Batch 0"):
        await generate_records(chat, people_schema)


async def test_empty_first_batch_cannot_seed_more(
    people_schema: dict[str, Any],
) -> None:
    chat = FakeChat(reply=lambda messages, schema: json.dumps({"records": []}))
    config = MockConfig(num_records_overall=20, records_per_batch=10)

    with pytest.raises(MockDataError, match="No records in 'records' to seed"):
        await generate_records(chat, people_schema, config)
