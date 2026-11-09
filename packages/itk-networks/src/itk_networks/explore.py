# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The attribute subgraph around one network, and its simplified form.

Graph logic only: colours, sizes and plot payloads belong to the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import polars as pl

from itk_networks.models import COUNT
from itk_networks.project import neighbors
from itk_networks.records import flag_totals
from itk_networks.text import (
    ATTRIBUTE_VALUE_SEPARATOR,
    ENTITY_LABEL,
    LIST_SEPARATOR,
    is_entity,
    node_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

Links = dict[str, frozenset[str]]

NODES_SCHEMA = {"node": pl.Utf8, "type": pl.Utf8, "flags": pl.Int64}
EDGES_SCHEMA = {"source": pl.Utf8, "target": pl.Utf8, "attribute": pl.Utf8}


def _add_attribute(subgraph: nx.Graph, node: str) -> None:
    subgraph.add_node(node, type=node_type(node), flags=0)


def _attribute_neighbors(
    graph: nx.Graph, links: Links, trimmed: frozenset[str], node: str
) -> list[str]:
    return sorted(
        n
        for n in neighbors(graph, links, node)
        if n not in trimmed and not is_entity(n)
    )


def network_subgraph(
    graph: nx.Graph,
    entities: Iterable[str],
    membership: dict[str, int] | None = None,
    flags: pl.DataFrame | None = None,
    trimmed: frozenset[str] = frozenset(),
    links: Links | None = None,
) -> nx.Graph:
    """Entities with their attributes, those attributes' attribute neighbours,
    and one further hop of attribute neighbours, skipping trimmed nodes.

    Nodes carry ``type``, ``flags`` and, for entities, ``network``.
    """
    membership = membership or {}
    links = links or {}
    subgraph = nx.Graph()
    for node in entities:
        network = str(membership[node]) if node in membership else ""
        subgraph.add_node(node, type=ENTITY_LABEL, network=network, flags=0)
        for first in sorted(neighbors(graph, links, node) - trimmed):
            if is_entity(first):
                if first not in subgraph:
                    other = str(membership[first]) if first in membership else ""
                    subgraph.add_node(first, type=ENTITY_LABEL, network=other, flags=0)
                subgraph.add_edge(node, first)
                continue
            _add_attribute(subgraph, first)
            subgraph.add_edge(node, first)
            for second in _attribute_neighbors(graph, links, trimmed, first):
                _add_attribute(subgraph, second)
                for third in _attribute_neighbors(graph, links, trimmed, second):
                    _add_attribute(subgraph, third)
                    subgraph.add_edge(second, third)
    if flags is not None and not flags.is_empty():
        for node, count in flag_totals(flags.filter(pl.col(COUNT) > 0)).items():
            if node in subgraph:
                subgraph.nodes[node]["flags"] = count
    return subgraph


def _parts(node: str) -> list[str]:
    return node.split(LIST_SEPARATOR)


def should_merge(first: str, second: str) -> bool:
    """Two attribute nodes merge when any part shares a type or a value."""
    return any(
        a.split(ATTRIBUTE_VALUE_SEPARATOR)[i] == b.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
        for i in range(2)
        for a in set(_parts(first))
        for b in set(_parts(second))
    )


def _merge(graph: nx.Graph, merge_list: list[str]) -> nx.Graph:
    graph = graph.copy()
    merged = LIST_SEPARATOR.join(sorted(merge_list))
    merged_type = LIST_SEPARATOR.join(
        sorted(graph.nodes[n]["type"] for n in merge_list)
    )
    merged_flags = max(graph.nodes[n].get("flags", 0) for n in merge_list)
    graph.add_node(merged, type=merged_type, flags=merged_flags)
    for node in merge_list:
        for neighbor in graph.neighbors(node):
            if neighbor not in merge_list:
                graph.add_edge(merged, neighbor)
        graph.remove_node(node)
    return graph


def merge_nodes(
    graph: nx.Graph, condition: Callable[[str, str], bool] = should_merge
) -> nx.Graph:
    for node in list(graph.nodes()):
        if node not in graph:
            continue
        merge_list = [node] + [
            n for n in list(graph.neighbors(node)) if n in graph and condition(node, n)
        ]
        if len(merge_list) > 1:
            graph = _merge(graph, merge_list)
    return graph


def _drop_leaf_attributes(graph: nx.Graph) -> None:
    for node in list(graph.nodes()):
        if graph.degree(node) < 2 and not is_entity(node):
            graph.remove_node(node)


def simplify(subgraph: nx.Graph) -> nx.Graph:
    """Drop single-degree attributes, merge related attribute neighbours, drop again."""
    graph = subgraph.copy()
    _drop_leaf_attributes(graph)
    graph = merge_nodes(graph)
    _drop_leaf_attributes(graph)
    return graph


def contains_entity(node: str) -> bool:
    """True when a (possibly merged) node includes an entity part."""
    return ENTITY_LABEL in node


def subgraph_frames(graph: nx.Graph) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``nodes(node, type, flags)`` and ``edges(source, target, attribute)``."""
    nodes = pl.DataFrame(
        {
            "node": list(graph.nodes()),
            "type": [d["type"] for _, d in graph.nodes(data=True)],
            "flags": [int(d.get("flags", 0)) for _, d in graph.nodes(data=True)],
        },
        schema=NODES_SCHEMA,
    )
    edges = pl.DataFrame(
        {
            "source": [s for s, _ in graph.edges()],
            "target": [t for _, t in graph.edges()],
            "attribute": [node_type(t) for _, t in graph.edges()],
        },
        schema=EDGES_SCHEMA,
    )
    return nodes, edges
