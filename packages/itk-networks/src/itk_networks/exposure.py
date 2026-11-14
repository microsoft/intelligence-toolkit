# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Flag exposure of one entity: shortest paths from every flagged entity to it."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx
import polars as pl

from itk_networks.explore import contains_entity
from itk_networks.models import (
    QUALIFIED_ENTITY,
    Exposure,
    ExposureNode,
    ExposurePath,
    ExposureSummary,
)
from itk_networks.records import flag_totals
from itk_networks.text import qualify_entity

if TYPE_CHECKING:
    from collections.abc import Iterable

Links = dict[str, frozenset[str]]


def exposure_nodes(
    entities: Iterable[str], graph: nx.Graph, links: Links | None = None
) -> list[str]:
    """Network entities plus every inferred-link endpoint present in ``graph``."""
    nodes = list(entities)
    present = set(nodes)
    for key, values in (links or {}).items():
        for node in (key, *sorted(values)):
            if node not in present and node in graph:
                nodes.append(node)
                present.add(node)
    return nodes


def _shortest_paths(graph: nx.Graph, source: str, target: str) -> list[list[str]]:
    if source not in graph or target not in graph:
        return []
    try:
        return [list(p) for p in nx.all_shortest_paths(graph, source, target)]
    except nx.NetworkXNoPath:
        return []


def exposure_data(
    flags: pl.DataFrame,
    entities: Iterable[str],
    entity: str,
    graph: nx.Graph,
    links: Links | None = None,
) -> Exposure:
    """Exposure of ``entity`` (unqualified) within the network subgraph.

    Paths from flagged entities that share a route into the selected entity are
    grouped into one :class:`ExposurePath` whose first step lists the sources.
    """
    target = qualify_entity(entity)
    nodes = exposure_nodes(entities, graph, links)
    if flags.is_empty():
        return Exposure(entity, ExposureSummary(0, 0, 0, 0), (), ())
    totals = flag_totals(flags.filter(pl.col(QUALIFIED_ENTITY).is_in(nodes)))
    flagged = sorted(node for node, count in totals.items() if count > 0)

    direct = totals.get(target, 0)
    total = sum(totals.values())
    related = len(flagged) - (1 if target in flagged else 0)

    annotated: list[ExposureNode] = []
    seen: set[ExposureNode] = set()
    routes: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for source in flagged:
        for path in _shortest_paths(graph, source, target):
            if len(path) <= 1:
                continue
            for step in path:
                if contains_entity(step):
                    count = totals.get(step, 0)
                    if count == 0:
                        continue
                    node = ExposureNode(step, flags=count)
                else:
                    node = ExposureNode(step, entities=graph.degree(step))
                if node not in seen:
                    seen.add(node)
                    annotated.append(node)
            routes[tuple(path[1:])].append(source)

    paths = tuple(
        ExposurePath((tuple(sorted(sources)), *((n,) for n in route)))
        for route, sources in routes.items()
    )
    summary = ExposureSummary(
        direct=direct, indirect=total - direct, paths=len(paths), entities=related
    )
    return Exposure(entity, summary, paths, tuple(annotated))


def _annotation(node: str, nodes: dict[str, ExposureNode]) -> str:
    found = nodes.get(node)
    if contains_entity(node):
        flags = found.flags if found is not None and found.flags is not None else 0
        return f"{node} [linked to {flags} flags]"
    entities = found.entities if found is not None else 0
    return f"{node} [linked to {entities} entities]"


def render_exposure(exposure: Exposure) -> str:
    """The legacy markdown: a summary line and one fenced block per path."""
    s = exposure.summary
    nodes = {n.node: n for n in exposure.nodes}
    text = "##### Flag Exposure Paths\n\n"
    text += (
        f"The selected entity **{exposure.entity}** has **{s.direct}** direct flags "
        f"and is linked to **{s.indirect}** indirect flags via **{s.paths}** paths "
        f"from **{s.entities}** related entities:\n\n"
    )
    for i, path in enumerate(exposure.paths):
        text += f"**Path {i + 1}**\n\n```\n"
        for ix, step in enumerate(path.steps):
            indent = "  " * ix
            for node in step:
                text += f"{indent}{_annotation(node, nodes)}\n"
            if ix < len(path.steps) - 1:
                text += f"{indent}--->\n"
        text += "```\n\n"
    return text
