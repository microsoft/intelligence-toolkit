# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Deterministic fakes, the shared test corpus and golden-file helpers.

The fakes mirror those of the v1 characterization tests
(``intelligence_toolkit/tests/characterization/query_text_data``) so v2 can be
compared with the v1 goldens. Regenerate v2 goldens with ``UPDATE_GOLDEN=1``
only when a behaviour change is intended, and review the diff.
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import itk_query_text_data as qtd
import pytest

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "corpus"
GOLDEN_DIR = HERE / "golden"
REPO_ROOT = HERE.parents[2]
V1_GOLDEN_DIR = (
    REPO_ROOT
    / "intelligence_toolkit"
    / "tests"
    / "characterization"
    / "query_text_data"
    / "golden"
)

QUERY = "What are the impacts of climate change?"
CORPUS_FILES = ["climate.txt", "health.txt", "economy.txt", "regions.csv"]
CHUNK_SIZE = 60
GRAPH_ARGS = {"max_cluster_size": 4, "min_edge_weight": 1, "min_node_degree": 1}
EMBEDDING_DIMS = 32
RELEVANT_TERMS = ("climate", "sea level", "carbon", "heat wave", "flood")


def search_config(analysis_update_interval: int) -> qtd.SearchConfig:
    return qtd.SearchConfig(
        adjacent_test_steps=1,
        community_relevance_tests=2,
        community_ranking_chunks=3,
        relevance_test_batch_size=2,
        relevance_test_budget=8,
        irrelevant_community_restart=2,
        analysis_update_interval=analysis_update_interval,
    )


def corpus_paths() -> list[Path]:
    return [CORPUS_DIR / name for name in CORPUS_FILES]


def _section(content: str, start: str, end: str | None = None) -> str:
    begin = content.index(start) + len(start)
    if end is None:
        return content[begin:].strip()
    return content[begin : content.index(end, begin)].strip()


class FakeLLM:
    """Routes completions by prompt type and returns deterministic responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._theme_points: dict[str, list[str]] = {}

    async def complete(
        self,
        messages,
        *,
        response_format=None,
        max_tokens=None,
        logit_bias=None,
        on_token=None,
    ) -> str:
        content = messages[0]["content"]
        kind = (
            response_format["json_schema"]["name"]
            if response_format
            else self._classify(content, max_tokens)
        )
        key, result = getattr(self, f"_{kind}")(content)
        self.calls.append({
            "kind": kind,
            "key": key,
            "stream": on_token is not None,
            "max_tokens": max_tokens,
            "logit_bias": logit_bias,
            "num_messages": len(messages),
        })
        if on_token is not None:
            on_token(result)
        return result

    @staticmethod
    def _classify(content: str, max_tokens) -> str:
        if max_tokens == 1:
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
            updates.append({
                "point_id": point_id,
                "point_title": f"Point from source {sid}",
                "source_ids": [sid],
            })
        themes = [
            {"theme_title": title, "point_ids": points}
            for title, points in sorted(self._theme_points.items())
        ]
        return ids, json.dumps({"updates": updates, "themes": themes})

    def _theme_summary(self, content):
        hint = _section(content, "--Theme hint--", "--Previous themes--")
        chunks = _section(content, "--Source text chunks--")
        sources = ", ".join(re.findall(r"'(\d+): ", chunks))
        return hint, json.dumps({
            "theme_title": f"Summary of {hint}",
            "theme_points": [
                {
                    "point_title": f"Key point for {hint}",
                    "point_evidence": f"Evidence drawn from sources [source: {sources}]",
                    "point_commentary": "These sources suggest a pattern.",
                }
            ],
        })

    def _final_report(self, content):
        theme_summaries = _section(content, "--Theme summaries--", "--User query--")
        cited = re.findall(r"\[source: ([^\]]+)\]", theme_summaries)
        ids = sorted({int(x) for x in re.findall(r"\d+", " ".join(cited))})
        return ids, json.dumps({
            "report_title": "Climate impacts report",
            "report_overview": "This report summarises the themes found.",
            "report_implications": "The implications are broad [source: 999].",
            "answer": f"Climate change drives coastal risk [source: {', '.join(map(str, ids[:2]))}].",
        })

    def _commentary(self, content):
        return None, "Commentary on the emerging themes [source: 1]."

    def _condense(self, content):
        return None, "Condensed answer."


def fake_vector(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMS
    for token in re.findall(r"[a-z]+", text.lower()):
        index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % EMBEDDING_DIMS
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed(self, texts):
        self.calls.append(len(texts))
        return [fake_vector(text) for text in texts]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, set | frozenset):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def load_v1_golden(name: str) -> Any:
    path = V1_GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"v1 golden {path.name} not available")
    return json.loads(path.read_text(encoding="utf-8"))


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
        pytest.fail(
            f"Missing golden {path.name}; run with UPDATE_GOLDEN=1 to create it."
        )
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected, f"Output differs from golden {path.name}"
