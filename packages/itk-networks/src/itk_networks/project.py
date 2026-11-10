# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Trim busy attributes and project the attribute graph onto entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import polars as pl

from itk_networks.models import TRIMMED_SCHEMA
from itk_networks.text import is_entity, node_type

if TYPE_CHECKING:
    from collections.abc import Iterable

Links = dict[str, frozenset[str]]


def trim_attributes(
    graph: nx.Graph,
    max_attribute_degree: int,
    removed_attributes: Iterable[str] = (),
) -> tuple[pl.DataFrame, frozenset[str]]:
    """Attributes shared by more than ``max_attribute_degree`` entities.

    Returns the frame of trimmed attributes with their degree, sorted by degree
    then name, and the full set of nodes to ignore (trimmed plus removed).
    """
    trimmed = sorted(
        (
            (node, degree)
            for node, degree in graph.degree()
            if not is_entity(node) and degree > max_attribute_degree
        ),
        key=lambda item: (item[1], item[0]),
    )
    frame = pl.DataFrame(
        {
            "Attribute": [node for node, _ in trimmed],
            "Linked Entities": [degree for _, degree in trimmed],
        },
        schema=TRIMMED_SCHEMA,
    )
    nodes = frozenset(node for node, _ in trimmed) | frozenset(removed_attributes)
    return frame, nodes


def neighbors(graph: nx.Graph, links: Links, node: str) -> set[str]:
    """Graph neighbours plus inferred links; ``links`` must be symmetric."""
    near = set(graph.neighbors(node)) if node in graph else set()
    near |= links.get(node, frozenset())
    near.discard(node)
    return near


def entity_neighbors(
    graph: nx.Graph, links: Links, trimmed: frozenset[str], node: str
) -> list[str]:
    if graph.number_of_nodes() == 0:
        return []
    if node not in graph:
        msg = f"Node {node} not in graph"
        raise ValueError(msg)
    return sorted(neighbors(graph, links, node) - trimmed)


def _usable(node: str, supporting: frozenset[str], trimmed: frozenset[str]) -> bool:
    """An attribute may carry a connection if it is neither supporting nor trimmed."""
    return bool(node) and node_type(node) not in supporting and node not in trimmed


def project_entity_graph(
    graph: nx.Graph,
    trimmed: frozenset[str] = frozenset(),
    links: Links | None = None,
    supporting_attribute_types: Iterable[str] = (),
) -> nx.Graph:
    """Entity-only graph: two entities are joined when they are reachable within
    three hops through usable attributes, equality edges or inferred links.

    Entities with no such connection do not appear in the projection.
    """
    links = links or {}
    supporting = frozenset(supporting_attribute_types)
    projected = nx.Graph()
    for node in (n for n in graph.nodes() if is_entity(n)):
        for first in entity_neighbors(graph, links, trimmed, node):
            if is_entity(first):
                projected.add_edge(node, first)
                continue
            if not _usable(first, supporting, trimmed):
                continue
            for second in neighbors(graph, links, first):
                if not _usable(second, supporting, trimmed):
                    continue
                if is_entity(second):
                    if second != node:
                        projected.add_edge(node, second)
                    continue
                for third in neighbors(graph, links, second):
                    if (
                        is_entity(third)
                        and third != node
                        and _usable(third, supporting, trimmed)
                    ):
                        projected.add_edge(node, third)
    return projected
