# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""End-to-end v2 workflow over the shared corpus, pinned to v2 golden snapshots."""

import asyncio

import itk_query_text_data as qtd
import pytest
from qtd2_support import (
    CHUNK_SIZE,
    GRAPH_ARGS,
    QUERY,
    FakeEmbedder,
    FakeLLM,
    assert_matches_golden,
    corpus_paths,
    search_config,
)


def run_flow(interval: int) -> dict:
    llm, embedder = FakeLLM(), FakeEmbedder()
    events: list = []
    commentary_tokens: list[str] = []
    condense_tokens: list[str] = []

    async def flow():
        corpus = qtd.load_documents(corpus_paths(), CHUNK_SIZE, on_event=events.append)
        graph = qtd.build_concept_graph(corpus, **GRAPH_ARGS, on_event=events.append)
        vectors = await qtd.embed_chunks(corpus, embedder, on_event=events.append)
        expanded = await qtd.rewrite_query(QUERY, graph, llm, top_concepts=10)
        result = await qtd.search(
            QUERY,
            corpus,
            graph,
            vectors,
            llm=llm,
            embedder=embedder,
            expanded_query=expanded,
            config=search_config(interval),
            on_event=events.append,
        )
        answer = await qtd.build_answer(result, corpus, llm)
        commentary = await qtd.generate_commentary(
            QUERY,
            dict(result.themes),
            result.analysis_markdown,
            corpus,
            llm,
            on_token=commentary_tokens.append,
        )
        condensed = await qtd.condense_answer(
            QUERY, answer, llm, on_token=condense_tokens.append
        )
        return graph, result, answer, commentary, condensed

    graph, result, answer, commentary, condensed = asyncio.run(flow())
    return {
        "communities": [
            [c.id, c.level, c.parent, c.concepts] for c in graph.communities
        ],
        "expanded_query": result.expanded_query,
        "tests": [[t.label, t.chunk_id, t.response] for t in result.tests],
        "relevant_chunk_ids": result.relevant_chunk_ids,
        "summary": result.summary,
        "themes": list(result.themes.items()),
        "analysis_markdown": result.analysis_markdown,
        "events": [_describe(event) for event in events],
        "answer": {
            "markdown": answer.markdown,
            "references": answer.references,
            "report": dict(answer.report),
            "themes": answer.themes,
        },
        "commentary": commentary,
        "commentary_tokens": commentary_tokens,
        "condensed": condensed,
        "condense_tokens": condense_tokens,
        "llm_calls": llm.calls,
        "embed_calls": embedder.calls,
    }


def _describe(event) -> list:
    if isinstance(event, qtd.Progress):
        return ["progress", event.step, event.current, event.total]
    if isinstance(event, qtd.ChunksTested):
        return ["tested", len(event.tests), event.relevant_chunk_ids]
    return ["analysis", sorted(event.structure.themes)]


@pytest.mark.parametrize("interval", [0, 3], ids=["no_live_analysis", "live_analysis"])
def test_query_flow(interval):
    assert_matches_golden(f"query_flow_interval_{interval}", run_flow(interval))


def test_flow_is_repeatable():
    assert run_flow(3) == run_flow(3)
