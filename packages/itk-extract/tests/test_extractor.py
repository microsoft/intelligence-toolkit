# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import json

import pytest
from itk_extract import ExtractConfig, Extractor, RecordSchema, from_texts

from .conftest import PEOPLE_SCHEMA, FakeBatchChat


async def test_end_to_end_merges_and_tabulates(
    fake_chat: FakeBatchChat, alice_text: str, bob_text: str
) -> None:
    extractor = Extractor(fake_chat)
    result = await extractor.extract(from_texts([alice_text, bob_text]), PEOPLE_SCHEMA)

    assert result.errors == []
    assert result.object["founded"] == 1999
    assert result.object["organisation"] == "ACME Ltd"
    assert [c.path for c in result.conflicts] == ["organisation"]

    people = result.frames["people"]
    assert people.get_column("name").to_list() == ["Alice", "Bob"]
    assert people.get_column("contact.email").to_list() == [
        "alice@acme.test",
        "bob@acme.test",
    ]
    assert people.get_column("source_id").to_list() == ["text-0", "text-1"]
    assert people.get_column("chunk_id").to_list() == ["text-0#0", "text-1#0"]

    events = result.frames["people.events"]
    assert events.height == 3
    assert "source_id" not in events.columns  # events were not top-level elements

    conflicts = result.conflicts_frame
    assert conflicts.get_column("path").to_list() == ["organisation"]
    assert conflicts.get_column("values").to_list() == [["Acme", "ACME Ltd"]]


async def test_accepts_record_schema_instance(
    fake_chat: FakeBatchChat, alice_text: str
) -> None:
    result = await Extractor(fake_chat).extract(
        from_texts([alice_text]), RecordSchema(PEOPLE_SCHEMA)
    )
    assert result.frames["people"].height == 1


async def test_chunking_uses_injected_splitter(
    fake_chat: FakeBatchChat, alice_text: str, bob_text: str
) -> None:
    def factory(chunk_tokens: int):
        assert chunk_tokens == 42
        return lambda text: text.split("\n---\n")

    extractor = Extractor(fake_chat, splitter_factory=factory)
    result = await extractor.extract(
        from_texts([f"{alice_text}\n---\n{bob_text}"]),
        PEOPLE_SCHEMA,
        ExtractConfig(chunk_tokens=42),
    )
    assert [e.chunk.id for e in result.extractions] == ["text-0#0", "text-0#1"]
    people = result.frames["people"]
    assert people.get_column("source_id").to_list() == ["text-0", "text-0"]
    assert people.get_column("chunk_id").to_list() == ["text-0#0", "text-0#1"]


async def test_bad_replies_are_recorded_not_raised(
    fake_chat: FakeBatchChat, alice_text: str
) -> None:
    result = await Extractor(fake_chat).extract(
        from_texts(["plain prose", alice_text]), PEOPLE_SCHEMA
    )
    assert result.errors_frame.get_column("chunk_id").to_list() == ["text-0#0"]
    assert result.frames["people"].height == 1


async def test_remerge_changes_policy_without_new_calls(
    fake_chat: FakeBatchChat, alice_text: str, bob_text: str
) -> None:
    extractor = Extractor(fake_chat)
    config = ExtractConfig()
    result = await extractor.extract(
        from_texts([alice_text, bob_text]), PEOPLE_SCHEMA, config
    )
    calls_before = len(fake_chat.calls)

    first = Extractor.remerge(
        result, config.model_copy(update={"merge_policy": "first"})
    )
    assert first.object["organisation"] == "Acme"
    plain = Extractor.remerge(
        result, config.model_copy(update={"annotate_provenance": False})
    )
    assert "source_id" not in plain.frames["people"].columns
    assert len(fake_chat.calls) == calls_before


async def test_max_sources_truncates(
    fake_chat: FakeBatchChat, alice_text: str, bob_text: str
) -> None:
    result = await Extractor(fake_chat).extract(
        from_texts([alice_text, bob_text]), PEOPLE_SCHEMA, ExtractConfig(max_sources=1)
    )
    assert result.frames["people"].get_column("name").to_list() == ["Alice"]


async def test_duplicate_source_ids_rejected(fake_chat: FakeBatchChat) -> None:
    sources = from_texts(["a"]) + from_texts(["b"])
    with pytest.raises(ValueError, match="Duplicate source id"):
        await Extractor(fake_chat).extract(sources, PEOPLE_SCHEMA)


async def test_validation_can_be_disabled(fake_chat: FakeBatchChat) -> None:
    loose = json.dumps({"organisation": 123, "people": []})
    strict = await Extractor(fake_chat).extract(from_texts([loose]), PEOPLE_SCHEMA)
    assert len(strict.errors) == 1
    relaxed = await Extractor(fake_chat).extract(
        from_texts([loose]), PEOPLE_SCHEMA, ExtractConfig(validate_output=False)
    )
    assert relaxed.errors == []
    assert relaxed.object["organisation"] == 123


async def test_no_sources(fake_chat: FakeBatchChat) -> None:
    result = await Extractor(fake_chat).extract([], PEOPLE_SCHEMA)
    assert result.object == {}
    assert result.frames["people"].is_empty()
    assert result.conflicts_frame.is_empty()
    assert fake_chat.calls == []
