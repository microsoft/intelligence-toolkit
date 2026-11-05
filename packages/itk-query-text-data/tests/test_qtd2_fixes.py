# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Regression tests for v1 defects that v2 fixes, plus edge cases v1 did not handle."""

import asyncio
import json
import math
import os
import subprocess
import sys

import itk_query_text_data as qtd
import pytest
from qtd2_support import (
    CHUNK_SIZE,
    GRAPH_ARGS,
    QUERY,
    REPO_ROOT,
    FakeEmbedder,
    FakeLLM,
    corpus_paths,
    search_config,
)


@pytest.fixture(scope="module")
def corpus():
    return qtd.load_documents(corpus_paths(), CHUNK_SIZE)


def test_neighbours_stay_within_a_source(corpus):
    for chunk in corpus:
        for neighbour in corpus.neighbours(chunk.id, 2):
            assert corpus[neighbour].source == chunk.source
    last_climate = max(c.id for c in corpus if c.source == "climate.txt")
    assert corpus.next_id(last_climate) is None
    assert corpus.neighbours(last_climate, 1) == {last_climate - 1}


@pytest.mark.parametrize(
    "path",
    [
        r"C:\data\reports\my file (1).txt",
        "/data/reports/my file (1).txt",
        "my file (1).txt",
    ],
)
def test_source_name_strips_directories_on_any_os(path):
    assert qtd.source_name(path) == "my_file_1.txt"


def test_unknown_citations_are_not_linked(corpus):
    text, references = qtd.link_sources(
        "Claim [source: 1, 999; 2] and [source: 0].", corpus
    )
    assert references == [1, 2]
    assert text == "Claim [source: [1](#source-1), [2](#source-2)] and."


class _BrokenThemeLLM(FakeLLM):
    def _theme_summary(self, content):
        key, result = super()._theme_summary(content)
        return key, ("not json" if key == "Even sources" else result)


def test_invalid_theme_summary_is_skipped(corpus):
    result = qtd.SearchResult(
        query=QUERY,
        expanded_query=QUERY,
        tests=(),
        relevant_chunk_ids=(1, 2),
        themes={"Even sources": (2,), "Odd sources": (1,)},
    )
    answer = asyncio.run(qtd.build_answer(result, corpus, _BrokenThemeLLM()))
    assert [theme["theme_title"] for theme in answer.themes] == [
        "Summary of Odd sources"
    ]
    assert answer.references == (1,)


def _run_search(corpus, config):
    llm, embedder = FakeLLM(), FakeEmbedder()
    graph = qtd.build_concept_graph(corpus, **GRAPH_ARGS)

    async def flow():
        vectors = await qtd.embed_chunks(corpus, embedder)
        result = await qtd.search(
            QUERY, corpus, graph, vectors, llm=llm, embedder=embedder, config=config
        )
        answer = await qtd.build_answer(result, corpus, llm)
        return graph, result, answer

    return asyncio.run(flow())


def test_corpus_without_concepts_still_searches():
    corpus = qtd.chunk_documents([
        qtd.Document("a.txt", "Climate. Floods."),
        qtd.Document("b.txt", "Rain. Snow."),
    ])
    graph, result, answer = _run_search(corpus, search_config(3))
    assert graph.communities == ()
    assert graph.max_level == -1
    assert [t.chunk_id for t in result.tests] == [1, 2]
    assert result.relevant_chunk_ids == (1,)
    assert answer.references == (1,)


def test_empty_corpus_returns_empty_result():
    corpus = qtd.chunk_documents([])
    _, result, answer = _run_search(corpus, search_config(0))
    assert result.tests == ()
    assert result.relevant_chunk_ids == ()
    assert dict(result.themes) == {}
    assert answer.references == ()


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("adjacent", [0, 1, 2])
def test_relevance_budget_is_never_exceeded(corpus, budget, adjacent):
    config = qtd.SearchConfig(
        relevance_test_budget=budget,
        relevance_test_batch_size=2,
        community_relevance_tests=2,
        community_ranking_chunks=3,
        irrelevant_community_restart=2,
        adjacent_test_steps=adjacent,
        analysis_update_interval=0,
    )
    _, result, _ = _run_search(corpus, config)
    assert 0 < len(result.tests) <= budget
    assert len({t.chunk_id for t in result.tests}) == len(result.tests)


@pytest.mark.parametrize(
    "overrides",
    [
        {"relevance_test_budget": 0},
        {"relevance_test_batch_size": 0},
        {"adjacent_test_steps": -1},
        {"analysis_update_interval": -1},
    ],
)
def test_search_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError, match="must be"):
        qtd.SearchConfig(**overrides)


def test_corpus_records_round_trip(corpus):
    records = corpus.to_records()
    records[0]["period"] = math.nan
    restored = qtd.Corpus.from_records(records)
    assert restored[1].period is None
    assert [c.to_prompt() for c in restored] == [c.to_prompt() for c in corpus]


def test_import_does_not_download_nltk_data():
    script = (
        "import nltk\n"
        "def fail(*args, **kwargs):\n"
        "    raise AssertionError('nltk.download called at import time')\n"
        "nltk.download = fail\n"
        "import itk_query_text_data\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=REPO_ROOT)


_SNAPSHOT_SCRIPT = """
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import itk_query_text_data as qtd
from qtd2_support import CHUNK_SIZE, GRAPH_ARGS, QUERY, FakeEmbedder, FakeLLM, corpus_paths, search_config
corpus = qtd.load_documents(corpus_paths(), CHUNK_SIZE)
graph = qtd.build_concept_graph(corpus, **GRAPH_ARGS)
async def flow():
    embedder = FakeEmbedder()
    vectors = await qtd.embed_chunks(corpus, embedder)
    return await qtd.search(QUERY, corpus, graph, vectors, llm=FakeLLM(), embedder=embedder,
                            config=search_config(3))
result = asyncio.run(flow())
print(json.dumps({
    "communities": [[c.id, list(c.concepts)] for c in graph.communities],
    "tests": [[t.label, t.chunk_id] for t in result.tests],
}))
"""


def test_results_do_not_depend_on_hash_seed():
    """v1 iterated over sets of strings, so its communities changed with PYTHONHASHSEED."""
    tests_dir = str(REPO_ROOT / "packages" / "itk-query-text-data" / "tests")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _SNAPSHOT_SCRIPT, tests_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        for seed in range(4)
    ]
    outputs = set()
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=300)
        assert proc.returncode == 0, stderr
        outputs.add(stdout.strip().splitlines()[-1])
    assert len(outputs) == 1
    assert json.loads(outputs.pop())["communities"]
