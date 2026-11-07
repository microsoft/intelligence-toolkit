# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from itk_extract.chunk import chunk_sources
from itk_extract.extract import extract_chunks, parse_reply
from itk_extract.models import ExtractConfig
from itk_extract.sources import from_texts

if TYPE_CHECKING:
    from itk_extract.schema import RecordSchema
    from itk_extract_testing import FakeBatchChat


def test_parse_reply_strips_fence(schema: RecordSchema) -> None:
    reply = '```json\n{"organisation": "Acme", "people": []}\n```'
    assert parse_reply(reply, schema) == {"organisation": "Acme", "people": []}


def test_parse_reply_rejects_bad_json(schema: RecordSchema) -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_reply("{oops", schema)


def test_parse_reply_rejects_non_object(schema: RecordSchema) -> None:
    with pytest.raises(TypeError, match="not an object"):
        parse_reply("[1, 2]", schema)


def test_parse_reply_validates_unless_told_not_to(schema: RecordSchema) -> None:
    bad = json.dumps({"organisation": 7, "people": []})
    with pytest.raises(ValueError, match="violates schema: organisation"):
        parse_reply(bad, schema)
    assert parse_reply(bad, schema, validate=False)["organisation"] == 7


def test_parse_reply_caps_reported_violations(schema: RecordSchema) -> None:
    bad = json.dumps({
        "organisation": 1,
        "founded": "x",
        "people": [{"name": 1, "role": 2, "contact": 3, "events": 4}],
    })
    with pytest.raises(ValueError, match="and 1 more"):
        parse_reply(bad, schema)


async def test_extract_chunks_one_call_per_distinct_text(
    schema: RecordSchema, fake_chat: FakeBatchChat, alice_text: str
) -> None:
    chunks = chunk_sources(from_texts([alice_text, alice_text]))
    extractions, errors = await extract_chunks(
        chunks, fake_chat, schema, ExtractConfig(guidance="Be brief.")
    )
    assert errors == []
    assert [e.chunk.id for e in extractions] == ["text-0#0", "text-1#0"]
    assert extractions[0].data["organisation"] == "Acme"
    assert len(fake_chat.calls) == 1
    assert len(fake_chat.calls[0]) == 1
    assert "Be brief." in fake_chat.calls[0][0][0]["content"]
    assert fake_chat.kwargs[0]["response_format"]["type"] == "json_schema"


async def test_extract_chunks_records_bad_replies(
    schema: RecordSchema, fake_chat: FakeBatchChat, alice_text: str
) -> None:
    chunks = chunk_sources(from_texts(["free text, not json", alice_text]))
    extractions, errors = await extract_chunks(
        chunks, fake_chat, schema, ExtractConfig()
    )
    assert [e.chunk.id for e in extractions] == ["text-1#0"]
    assert [e.chunk.id for e in errors] == ["text-0#0"]
    assert "not valid JSON" in errors[0].message
    assert errors[0].raw == "not json at all"


async def test_extract_chunks_fail_fast(
    schema: RecordSchema, fake_chat: FakeBatchChat
) -> None:
    chunks = chunk_sources(from_texts(["free text"]))
    with pytest.raises(ValueError, match="Chunk text-0#0"):
        await extract_chunks(chunks, fake_chat, schema, ExtractConfig(fail_fast=True))


async def test_extract_chunks_empty_input(
    schema: RecordSchema, fake_chat: FakeBatchChat
) -> None:
    assert await extract_chunks([], fake_chat, schema, ExtractConfig()) == ([], [])
    assert fake_chat.calls == []


async def test_extract_chunks_reply_count_mismatch(schema: RecordSchema) -> None:
    async def short_chat(messages_list, callbacks=None, **kwargs):
        await asyncio.sleep(0)
        return []

    chunks = chunk_sources(from_texts(["x"]))
    with pytest.raises(ValueError, match="0 replies for 1 prompts"):
        await extract_chunks(chunks, short_chat, schema, ExtractConfig())
