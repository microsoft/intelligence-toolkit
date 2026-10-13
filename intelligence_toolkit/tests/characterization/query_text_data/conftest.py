# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Deterministic fakes and golden-file helpers for query_text_data characterization tests.

These tests pin the observable behaviour of the v1 workflow so a v2 rewrite can be
checked against it. Regenerate goldens with ``UPDATE_GOLDEN=1`` only when a behaviour
change is intended, and review the diff.
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest

from intelligence_toolkit.AI.base_embedder import BaseEmbedder
from intelligence_toolkit.AI.client import OpenAIClient

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "corpus"
GOLDEN_DIR = HERE / "golden"

EMBEDDING_DIMS = 32
RELEVANT_TERMS = ("climate", "sea level", "carbon", "heat wave", "flood")


def _section(content: str, start: str, end: str | None = None) -> str:
    begin = content.index(start) + len(start)
    if end is None:
        return content[begin:].strip()
    return content[begin : content.index(end, begin)].strip()


class FakeLLM:
    """Routes chat calls by prompt type and returns deterministic responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._theme_points: dict[str, list[str]] = {}

    def respond(self, messages, stream, callbacks, kwargs) -> str:
        content = messages[0]["content"]
        response_format = kwargs.get("response_format")
        kind = (
            response_format["json_schema"]["name"]
            if response_format
            else self._classify(content, kwargs)
        )
        handler = getattr(self, f"_{kind}")
        key, result = handler(content)
        self.calls.append(
            {
                "kind": kind,
                "key": key,
                "stream": stream,
                "max_tokens": kwargs.get("max_tokens"),
                "logit_bias": kwargs.get("logit_bias"),
                "num_messages": len(messages),
            }
        )
        for callback in callbacks or []:
            callback.on_llm_new_token(result)
        return result

    @staticmethod
    def _classify(content: str, kwargs) -> str:
        if kwargs.get("max_tokens") == 1:
            return "relevance"
        if "rewriting a user query" in content:
            return "query_anchoring"
        if "providing commentary on a set of themes" in content:
            return "commentary"
        if "generating an output from an extended report" in content:
            return "condense"
        msg = f"Unexpected prompt: {content[:200]}"
        raise AssertionError(msg)

    def _relevance(self, content):
        chunk = json.loads(_section(content, "Text chunk:\n\n", "\n\nUser query:"))
        text = chunk["text_chunk"].lower()
        key = f"{chunk['title']}#{chunk['chunk_id']}"
        if "flood" in text:
            # Exercises the "yes." -> "Yes" normalisation in process_relevance_responses.
            return key, "yes."
        return key, "Yes" if any(term in text for term in RELEVANT_TERMS) else "No"

    def _query_anchoring(self, content):
        query = _section(content, "User query:\n\n", "\n\nData concepts:")
        concepts = _section(content, "Data concepts:\n\n", "\n\nOutput query:")
        top = ", ".join(concepts.split(", ")[:5])
        return query, f"{query} (anchored to: {top})"

    def _thematic_analysis(self, content):
        sources = _section(
            content, "--New sources by source ID--", "--Output JSON object--"
        )
        ids = [int(x) for x in re.findall(r"^(\d+):$", sources, re.MULTILINE)]
        updates = []
        for sid in ids:
            point_id = f"P{sid}"
            theme = "Odd sources" if sid % 2 else "Even sources"
            if point_id not in self._theme_points.setdefault(theme, []):
                self._theme_points[theme].append(point_id)
            updates.append(
                {"point_id": point_id, "point_title": f"Point from source {sid}", "source_ids": [sid]}
            )
        themes = [
            {"theme_title": title, "point_ids": points}
            for title, points in sorted(self._theme_points.items())
        ]
        return ids, json.dumps({"updates": updates, "themes": themes})

    def _theme_summary(self, content):
        hint = _section(content, "--Theme hint--", "--Previous themes--")
        chunks = _section(content, "--Source text chunks--")
        ids = re.findall(r"'(\d+): ", chunks)
        sources = ", ".join(ids)
        return hint, json.dumps(
            {
                "theme_title": f"Summary of {hint}",
                "theme_points": [
                    {
                        "point_title": f"Key point for {hint}",
                        # No "**Source evidence**:" prefix, to exercise _ensure_theme_formatting.
                        "point_evidence": f"Evidence drawn from sources [source: {sources}]",
                        "point_commentary": "These sources suggest a pattern.",
                    }
                ],
            }
        )

    def _final_report(self, content):
        theme_summaries = _section(content, "--Theme summaries--", "--User query--")
        ids = sorted(
            {int(x) for x in re.findall(r"\d+", " ".join(re.findall(r"\[source: ([^\]]+)\]", theme_summaries)))}
        )
        cited = ", ".join(str(x) for x in ids[:2])
        return ids, json.dumps(
            {
                "report_title": "Climate impacts report",
                "report_overview": "This report summarises the themes found.",
                "report_implications": "The implications are broad [source: 999].",
                "answer": f"Climate change drives coastal risk [source: {cited}].",
            }
        )

    def _commentary(self, content):
        return None, "Commentary on the emerging themes [source: 1]."

    def _condense(self, content):
        return None, "Condensed answer."


@pytest.fixture()
def fake_llm(monkeypatch):
    llm = FakeLLM()

    def generate_chat(self, messages, stream=True, callbacks=None, **kwargs):
        return llm.respond(messages, stream, callbacks, kwargs)

    async def generate_chat_async(self, messages, stream=True, callbacks=None, **kwargs):
        return llm.respond(messages, stream, callbacks, kwargs)

    monkeypatch.setattr(OpenAIClient, "generate_chat", generate_chat)
    monkeypatch.setattr(OpenAIClient, "generate_chat_async", generate_chat_async)
    # Clients are constructed per call; avoid needing real credentials or SDK clients.
    monkeypatch.setattr(OpenAIClient, "__init__", lambda self, *a, **k: None)
    return llm


def fake_vector(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMS
    for token in re.findall(r"[a-z]+", text.lower()):
        index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % EMBEDDING_DIMS
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbedder(BaseEmbedder):
    def _generate_embedding(self, text: str) -> list[float]:
        return fake_vector(text)

    async def _generate_embedding_async(self, text: str) -> list[float]:
        return fake_vector(text)


@pytest.fixture()
def fake_embedder(tmp_path):
    return FakeEmbedder(db_path=str(tmp_path / "embeddings"), check_token_count=False)


class RecordingCallback:
    def __init__(self) -> None:
        self.batches: list[tuple[int, int]] = []
        self.tokens: list[str] = []

    def on_batch_change(self, current: int, total: int) -> None:
        self.batches.append((current, total))

    def on_llm_new_token(self, token: str) -> None:
        self.tokens.append(token)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, set | frozenset):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def assert_matches_golden(name: str, data: Any) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    actual = to_jsonable(data)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return
    if not path.exists():
        pytest.fail(f"Missing golden {path.name}; run with UPDATE_GOLDEN=1 to create it.")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected, f"Output differs from golden {path.name}"
