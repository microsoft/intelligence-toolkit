# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Noun-phrase concept extraction, co-occurrence graphs and hierarchical communities."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from functools import cache

import networkx as nx
import numpy as np

from itk_query_text_data.models import (
    HUB_NODE,
    Community,
    ConceptGraph,
    Corpus,
    Progress,
)

EventHandler = Callable[[Progress], None]

_NLTK_RESOURCES = {
    "brown": "corpora/brown",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}
_CONCEPT_WORD = re.compile(r"^[a-zA-Z0-9\-]+\n?$")
_STD_TRIM = 3
_LEIDEN_SEED = 42


@cache
def ensure_nltk_data() -> None:
    """Download the NLTK data TextBlob needs, once per process, only if missing."""
    import nltk

    for package, resource in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def extract_concepts(text: str) -> list[str]:
    """Multi-word noun phrases made only of letters, digits and hyphens, sorted and unique."""
    from textblob import TextBlob

    ensure_nltk_data()
    concepts = set()
    for phrase in TextBlob(text).noun_phrases:
        parts = [part for part in re.split(r"\s+", phrase) if part]
        if len(parts) > 1 and all(_CONCEPT_WORD.match(part) for part in parts):
            concepts.add(phrase.replace("\n", ""))
    return sorted(concepts)


def build_concept_graph(
    corpus: Corpus,
    *,
    max_cluster_size: int = 25,
    min_edge_weight: int = 2,
    min_node_degree: int = 2,
    on_event: EventHandler | None = None,
) -> ConceptGraph:
    """Extract concepts per chunk, link co-occurring concepts and detect communities.

    Concepts with an outlying degree (above mean + 3 std) are dropped as too
    generic, then weak edges and weakly connected concepts are pruned before
    hierarchical Leiden community detection.
    """
    chunk_concepts: dict[int, tuple[str, ...]] = {}
    concept_chunks: dict[str, list[int]] = defaultdict(list)
    node_counts: dict[str, dict[str | None, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    edge_counts: dict[tuple[str, str], dict[str | None, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for position, chunk in enumerate(corpus, start=1):
        if on_event is not None:
            on_event(Progress("concepts", position, len(corpus)))
        concepts = extract_concepts(chunk.to_prompt())
        chunk_concepts[chunk.id] = tuple(concepts)
        for concept in concepts:
            concept_chunks[concept].append(chunk.id)
        scopes = (None,) if chunk.period is None else (None, chunk.period)
        for scope in scopes:
            for i, first in enumerate(concepts):
                node_counts[first][scope] += 1
                for second in concepts[i + 1 :]:
                    edge_counts[first, second][scope] += 1

    graphs: dict[str | None, nx.Graph] = defaultdict(nx.Graph)
    graphs[None] = nx.Graph()
    for node, counts in node_counts.items():
        for scope, count in counts.items():
            graphs[scope].add_node(node, count=count)
    for (first, second), counts in edge_counts.items():
        for scope, count in counts.items():
            graphs[scope].add_edge(first, second, weight=count)

    graph = graphs.pop(None)
    _prune(graph, min_edge_weight, min_node_degree)
    communities = _detect_communities(graph, max_cluster_size)

    final_community = {}
    for community in sorted(communities, key=lambda c: c.level):
        for concept in community.concepts:
            final_community[concept] = community.id
    for node, data in graph.nodes(data=True):
        data["community"] = final_community.get(node)
    period_graphs = {}
    for period, period_graph in graphs.items():
        period_graph.remove_nodes_from([
            n for n in list(period_graph) if n not in graph
        ])
        for node, data in period_graph.nodes(data=True):
            data["community"] = final_community.get(node)
        period_graphs[period] = period_graph

    return ConceptGraph(
        graph=graph,
        communities=tuple(communities),
        chunk_concepts=chunk_concepts,
        concept_chunks={c: tuple(ids) for c, ids in concept_chunks.items()},
        period_graphs=period_graphs,
    )


def _prune(graph: nx.Graph, min_edge_weight: int, min_node_degree: int) -> None:
    if graph.number_of_nodes() == 0:
        return
    degrees = [degree for _, degree in graph.degree()]
    cutoff = np.mean(degrees) + _STD_TRIM * np.std(degrees)
    graph.remove_nodes_from([n for n, d in graph.degree() if d > cutoff])
    graph.remove_edges_from([
        (u, v) for u, v, w in graph.edges(data="weight") if w < min_edge_weight
    ])
    graph.remove_nodes_from([n for n, d in graph.degree() if d < min_node_degree])


def _leiden_edges(graph: nx.Graph) -> list[tuple[str, str, float]]:
    """Graph edges plus a hub linking each component's best-connected concept.

    The hub lets Leiden partition disconnected components together; ties are
    broken by name so results do not depend on set iteration order.
    """
    linked = graph.copy()
    for component in list(nx.connected_components(graph)):
        top = max(sorted(component), key=graph.degree)
        linked.add_edge(top, HUB_NODE, weight=1)
    return [(str(u), str(v), float(w)) for u, v, w in linked.edges(data="weight")]


def _detect_communities(graph: nx.Graph, max_cluster_size: int) -> list[Community]:
    if graph.number_of_nodes() == 0:
        return []
    from graspologic_native import hierarchical_leiden

    assignments = hierarchical_leiden(
        edges=_leiden_edges(graph),
        starting_communities=None,
        resolution=1.0,
        randomness=0.001,
        iterations=1,
        use_modularity=True,
        max_cluster_size=max_cluster_size,
        seed=_LEIDEN_SEED,
    )
    members: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    sizes: dict[int, int] = defaultdict(int)
    parents: dict[int, int | None] = {}
    for entry in assignments:
        sizes[entry.cluster] += 1
        if entry.node == HUB_NODE:
            members[entry.level].setdefault(entry.cluster, [])
        else:
            members[entry.level][entry.cluster].append(entry.node)
        parents[entry.cluster] = entry.parent_cluster

    labels: dict[int, str] = {}
    communities: list[Community] = []
    for level in sorted(members):
        siblings: dict[int | None, list[int]] = defaultdict(list)
        for cluster, concepts in members[level].items():
            if concepts:
                siblings[parents[cluster] if level > 0 else None].append(cluster)
        for parent, clusters in siblings.items():
            prefix = "1" if parent is None else labels[parent]
            # Sizes include the hub so labels match v1's size ranking.
            ranked = sorted(clusters, key=lambda c: -sizes[c])
            for rank, cluster in enumerate(ranked, start=1):
                labels[cluster] = f"{prefix}.{rank}"
                communities.append(
                    Community(
                        id=labels[cluster],
                        level=level,
                        parent=None if parent is None else labels[parent],
                        concepts=tuple(sorted(members[level][cluster])),
                    )
                )
    return communities
