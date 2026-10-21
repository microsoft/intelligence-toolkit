# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""v2 must reproduce v1's data preparation and search over the same inputs.

The expected values come from the v1 characterization goldens. Known, intended
divergences are covered in ``test_qtd2_fixes.py``.
"""

import asyncio
import json

import itk_query_text_data as qtd
import pytest
from itk_query_text_data.models import HUB_NODE
from qtd2_support import (
    CHUNK_SIZE,
    GRAPH_ARGS,
    QUERY,
    FakeEmbedder,
    FakeLLM,
    corpus_paths,
    load_v1_golden,
    search_config,
    to_jsonable,
)


@pytest.fixture(scope="module")
def corpus():
    return qtd.load_documents(corpus_paths(), CHUNK_SIZE)


@pytest.fixture(scope="module")
def graph(corpus):
    return qtd.build_concept_graph(corpus, **GRAPH_ARGS)


def test_chunks_match_v1(corpus):
    golden = load_v1_golden("chunking")
    assert list(corpus.sources) == golden["label_order"]
    by_source: dict[str, list] = {}
    for chunk in corpus:
        by_source.setdefault(chunk.source, []).append(json.loads(chunk.to_prompt()))
    assert to_jsonable(by_source) == golden["label_to_chunks"]


def test_concepts_match_v1(graph):
    golden = load_v1_golden("processing")
    assert to_jsonable(dict(graph.chunk_concepts)) == golden["cid_to_concepts"]
    assert to_jsonable(dict(graph.concept_chunks)) == golden["concept_to_cids"]


def test_graphs_match_v1(graph):
    golden = load_v1_golden("processing")["graphs"]

    def shape(g):
        return {
            "nodes": sorted(
                [n, d["count"]] for n, d in g.nodes(data=True) if n != HUB_NODE
            ),
            "edges": sorted(
                [*sorted((u, v)), w]
                for u, v, w in g.edges(data="weight")
                if HUB_NODE not in (u, v)
            ),
        }

    def golden_shape(g):
        return {
            "nodes": sorted([n, c] for n, c, _ in g["nodes"] if n != HUB_NODE),
            "edges": sorted(e for e in g["edges"] if HUB_NODE not in e[:2]),
        }

    assert shape(graph.graph) == golden_shape(golden.pop("ALL"))
    assert {p: shape(g) for p, g in graph.period_graphs.items()} == {
        p: golden_shape(g) for p, g in golden.items()
    }


def test_communities_match_v1(graph):
    golden = load_v1_golden("processing")
    expected = sorted(
        [node, level, label, parent]
        for node, level, label, parent, _ in golden["communities"]
        if node != HUB_NODE
    )
    actual = sorted(
        [concept, c.level, c.id, c.parent]
        for c in graph.communities
        for concept in c.concepts
    )
    assert actual == expected
    final = {
        node: label for node, _, label, _, is_final in golden["communities"] if is_final
    }
    assert {n: d["community"] for n, d in graph.graph.nodes(data=True)} == {
        n: label for n, label in final.items() if n != HUB_NODE
    }


def test_community_networks_match_v1(graph):
    golden = load_v1_golden("community_graph")

    def members(level_networks):
        return sorted(
            [sorted(net["nodes"]), sorted(list(e) for e in net["edges"])]
            for net in level_networks.values()
        )

    actual = {
        str(level): {
            label: {
                "nodes": sorted(g.nodes()),
                "edges": sorted(sorted(e) for e in g.edges()),
            }
            for label, g in networks.items()
        }
        for level, networks in graph.community_networks().items()
    }
    assert {lvl: members(n) for lvl, n in actual.items()} == {
        lvl: members(n) for lvl, n in golden.items()
    }


def test_embeddings_match_v1(corpus):
    golden = load_v1_golden("embedding")
    vectors = asyncio.run(qtd.embed_chunks(corpus, FakeEmbedder()))
    assert to_jsonable({cid: v[:4] for cid, v in vectors.items()}) == golden


@pytest.mark.parametrize("interval", [0, 3])
def test_search_matches_v1(corpus, graph, interval):
    golden = load_v1_golden(f"query_flow_interval_{interval}")
    llm, embedder = FakeLLM(), FakeEmbedder()

    async def flow():
        vectors = await qtd.embed_chunks(corpus, embedder)
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
        )
        return expanded, result

    expanded, result = asyncio.run(flow())
    assert expanded == golden["expanded_query"]
    assert list(result.relevant_chunk_ids) == golden["relevant_cids"]
    assert to_jsonable(list(result.themes.items())) == golden["clustered_cids"]
    history = [[t.label, t.chunk_id, t.response] for t in result.tests]
    # v1 neighbours cross file boundaries, so it under-counts the neighbour budget and
    # spends the last two tests on topic 1.2; v2 tests same-source neighbours instead.
    assert history[:6] == golden["test_history"][:6]
    if interval:
        structure = result.analysis
        assert (
            to_jsonable({
                "points": dict(structure.points),
                "point_sources": dict(structure.point_sources),
                "themes": dict(structure.themes),
            })
            == golden["commentary_structure"]
        )
