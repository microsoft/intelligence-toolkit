import json
from types import SimpleNamespace

import pytest

from intelligence_toolkit.query_text_data import answer_builder


def test_select_representative_cids_prefers_central_vectors():
    cids = [1, 2, 3, 4]
    cid_to_vector = {
        1: [0.0],
        2: [1.0],
        3: [9.0],
        4: [10.0],
    }

    selected = answer_builder.select_representative_cids(
        cids,
        cid_to_vector,
        target_count=2,
        min_retention_ratio=0.0,
    )

    assert selected == [2, 3]


def test_select_representative_cids_when_vectors_missing_returns_slice():
    cids = [10, 11, 12]
    cid_to_vector = {}

    selected = answer_builder.select_representative_cids(
        cids,
        cid_to_vector,
        target_count=2,
        min_retention_ratio=0.0,
    )

    assert selected == [10, 11]


def test_select_representative_cids_with_zero_target_returns_empty():
    cids = [1, 2, 3]
    cid_to_vector = {cid: [float(cid)] for cid in cids}

    selected = answer_builder.select_representative_cids(
        cids,
        cid_to_vector,
        target_count=0,
        min_retention_ratio=0.0,
    )

    assert selected == []


def test_select_representative_cids_respects_minimum_ratio():
    cids = list(range(10))
    cid_to_vector = {cid: [float(cid)] for cid in cids}

    selected = answer_builder.select_representative_cids(
        cids,
        cid_to_vector,
        target_count=2,
        min_retention_ratio=0.5,
    )

    assert len(selected) == 5


@pytest.mark.asyncio
async def test_answer_query_limits_chunks_and_preserves_theme_names(monkeypatch):
    captured_theme_messages = []

    def fake_prepare_messages(prompt, variables, *args, **kwargs):
        if "theme" in variables:
            captured_theme_messages.append(variables)
            return {
                "theme": variables["theme"],
                "chunks": variables["chunks"],
                "query": variables["query"],
            }
        return {"content": variables["content"], "query": variables["query"]}

    async def fake_map_generate_text(ai_configuration, messages, response_format):
        themes = []
        for message in messages:
            themes.append(message["theme"])
        return [
            json.dumps({"theme_title": theme, "theme_points": []})
            for theme in themes
        ]

    def fake_generate_text(ai_configuration, messages, response_format):
        return json.dumps(
            {
                "answer": "mock answer",
                "report_title": "mock report",
                "report_overview": "mock overview",
                "report_implications": "mock implications",
            }
        )

    monkeypatch.setattr(answer_builder.utils, "prepare_messages", fake_prepare_messages)
    monkeypatch.setattr(answer_builder.utils, "map_generate_text", fake_map_generate_text)
    monkeypatch.setattr(answer_builder.utils, "generate_text", fake_generate_text)

    cid_to_text = {
        cid: json.dumps({
            "title": f"Doc {cid}",
            "chunk_id": cid,
            "text_chunk": f"Chunk text {cid}",
        })
        for cid in range(1, 7)
    }
    processed_chunks = SimpleNamespace(cid_to_text=cid_to_text)

    clustered_cids = {
        "Theme A": [1, 2, 3, 4, 5],
        "Theme B": [6],
    }
    cid_to_vector = {
        cid: [float(cid)]
        for cid in range(1, 7)
    }

    result = await answer_builder.answer_query(
        ai_configuration=None,
        query="Sample query",
        expanded_query="Expanded sample query",
        processed_chunks=processed_chunks,
        clustered_cids=clustered_cids,
        cid_to_vector=cid_to_vector,
    max_chunks_per_theme=2,
        min_chunk_retention_ratio=0.0,
    )

    assert {variables["theme"] for variables in captured_theme_messages} == {"Theme A", "Theme B"}
    assert all(len(variables["chunks"]) <= 2 for variables in captured_theme_messages)
    assert "topic" not in result.extended_answer.lower()


@pytest.mark.asyncio
async def test_answer_query_retains_minimum_ratio(monkeypatch):
    chunk_lengths = []

    def fake_prepare_messages(prompt, variables, *args, **kwargs):
        chunk_lengths.append(len(variables.get("chunks", [])))
        return variables

    async def fake_map_generate_text(ai_configuration, messages, response_format):
        return [json.dumps({"theme_title": m["theme"], "theme_points": []}) for m in messages]

    def fake_generate_text(ai_configuration, messages, response_format):
        return json.dumps(
            {
                "answer": "mock answer",
                "report_title": "mock report",
                "report_overview": "mock overview",
                "report_implications": "mock implications",
            }
        )

    monkeypatch.setattr(answer_builder.utils, "prepare_messages", fake_prepare_messages)
    monkeypatch.setattr(answer_builder.utils, "map_generate_text", fake_map_generate_text)
    monkeypatch.setattr(answer_builder.utils, "generate_text", fake_generate_text)

    cid_to_text = {cid: json.dumps({"title": str(cid), "chunk_id": cid, "text_chunk": "x"}) for cid in range(10)}
    processed_chunks = SimpleNamespace(cid_to_text=cid_to_text)
    clustered_cids = {"Theme": list(range(10))}
    cid_to_vector = {cid: [float(cid)] for cid in range(10)}

    await answer_builder.answer_query(
        ai_configuration=None,
        query="q",
        expanded_query="eq",
        processed_chunks=processed_chunks,
        clustered_cids=clustered_cids,
        cid_to_vector=cid_to_vector,
    max_chunks_per_theme=2,
        min_chunk_retention_ratio=0.6,
    )

    assert chunk_lengths and chunk_lengths[0] == 6
