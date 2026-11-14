# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Generate records: one unseeded batch, then rounds of seeded batches."""

from __future__ import annotations

import asyncio
import json
import math
import random
from typing import TYPE_CHECKING, Any

from itk_mock.frames import records_at, to_frames
from itk_mock.models import MockConfig, MockDataError, MockResult
from itk_mock.prompts import seeded_messages, unseeded_messages
from itk_mock.schema import extract_array_fields

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import polars as pl

    from itk_mock.chat import ChatFn


def parse_reply(text: str, batch: int) -> dict[str, Any]:
    """Parse one model reply into a JSON object."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        msg = f"Batch {batch}: AI did not return a valid JSON response. {e}"
        raise MockDataError(msg) from e
    if not isinstance(data, dict):
        msg = f"Batch {batch}: AI returned a JSON {type(data).__name__}, not an object"
        raise MockDataError(msg)
    return data


def merge_json_objects(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Merge two objects: arrays concatenate, objects merge, scalars take the second.

    Returns the merged object and the dotted paths of scalars that differed.
    Keys keep the order in which they were first seen.
    """
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for key in [*first, *(k for k in second if k not in first)]:
        if key not in second:
            merged[key] = first[key]
        elif key not in first:
            merged[key] = second[key]
        else:
            a, b = first[key], second[key]
            if isinstance(a, dict) and isinstance(b, dict):
                merged[key], nested = merge_json_objects(a, b)
                conflicts.extend(f"{key}.{path}" for path in nested)
            elif isinstance(a, list) and isinstance(b, list):
                merged[key] = a + b
            else:
                if a != b:
                    conflicts.append(key)
                merged[key] = b
    return merged, conflicts


def merge_objects(
    objects: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Fold ``merge_json_objects`` over generated objects in order."""
    merged: dict[str, Any] = dict(objects[0]) if objects else {}
    conflicts: list[str] = []
    for obj in objects[1:]:
        merged, new_conflicts = merge_json_objects(merged, obj)
        conflicts.extend(new_conflicts)
    return merged, conflicts


def sample_from_record_array(
    current_object: Any, record_array: Sequence[str], k: int, rng: random.Random
) -> list[Any]:
    """Up to ``k`` distinct records of the array, or all of them if fewer."""
    records = records_at(current_object, record_array)
    return rng.sample(records, k) if len(records) > k else records


def select_random_records(
    num_records: int, category_to_count: Mapping[str, int], rng: random.Random
) -> dict[str, list[int]]:
    """Disjoint random record indices for each category."""
    selected = rng.sample(range(num_records), sum(category_to_count.values()))
    category_to_ids: dict[str, list[int]] = {}
    for category, count in category_to_count.items():
        category_to_ids[category] = selected[:count]
        selected = selected[count:]
    return category_to_ids


async def generate_unseeded(
    chat: ChatFn,
    schema: dict[str, Any],
    primary_record_array: list[str],
    config: MockConfig,
) -> str:
    """One batch of unrelated records."""
    messages = unseeded_messages(
        config.generation_guidance, primary_record_array, config.records_per_batch
    )
    return await chat(messages, response_schema=schema, temperature=config.temperature)


async def generate_seeded(
    chat: ChatFn,
    schema: dict[str, Any],
    primary_record_array: list[str],
    seed_records: Sequence[Any],
    config: MockConfig,
) -> list[str]:
    """One batch per seed record, generated concurrently, in seed order."""
    calls = [
        chat(
            seeded_messages(
                seed_record,
                config.generation_guidance,
                primary_record_array,
                config.records_per_batch,
                config.duplicate_records_per_batch,
                config.related_records_per_batch,
            ),
            response_schema=schema,
            temperature=config.temperature,
        )
        for seed_record in seed_records
    ]
    return list(await asyncio.gather(*calls))


async def generate_records(
    chat: ChatFn,
    schema: dict[str, Any],
    config: MockConfig | None = None,
    rng: random.Random | None = None,
    on_update: Callable[[dict[str, pl.DataFrame]], None] | None = None,
) -> MockResult:
    """Generate ``config.num_records_overall`` primary records.

    The first batch is unseeded. Each later round samples up to
    ``parallel_batches`` existing primary records with ``rng`` (default
    ``random.Random(config.seed)``) and generates one batch per seed. Record
    counts are what the model was asked for; replies are not truncated.
    ``on_update`` receives the frames after every round.
    """
    config = config or MockConfig()
    rng = rng or random.Random(config.seed)
    record_arrays = extract_array_fields(schema)
    if not record_arrays:
        msg = "Schema has no array field to generate records for"
        raise MockDataError(msg)
    primary = record_arrays[0]

    reply = await generate_unseeded(chat, schema, primary, config)
    generated = [parse_reply(reply, 0)]
    current = dict(generated[0])
    conflicts: list[str] = []
    if on_update is not None:
        on_update(to_frames(current, record_arrays))

    requested = config.records_per_batch
    while requested < config.num_records_overall:
        remaining = config.num_records_overall - requested
        batches = min(
            config.parallel_batches, math.ceil(remaining / config.records_per_batch)
        )
        seeds = sample_from_record_array(current, primary, batches, rng)
        if not seeds:
            msg = f"No records in {'.'.join(primary)!r} to seed further batches from"
            raise MockDataError(msg)
        replies = await generate_seeded(chat, schema, primary, seeds, config)
        for reply in replies:
            new_object = parse_reply(reply, len(generated))
            generated.append(new_object)
            current, new_conflicts = merge_json_objects(current, new_object)
            conflicts.extend(new_conflicts)
        requested += config.records_per_batch * len(seeds)
        if on_update is not None:
            on_update(to_frames(current, record_arrays))

    return MockResult(
        schema=schema,
        config=config,
        record_arrays=record_arrays,
        generated_objects=generated,
        json_object=current,
        conflicts=conflicts,
        frames=to_frames(current, record_arrays),
    )
