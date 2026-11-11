# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""One weighted attribute co-occurrence graph per period."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

import networkx as nx
import polars as pl

from itk_patterns.models import (
    EDGE_SCHEMA,
    FULL_ATTRIBUTE,
    PERIOD,
    SOURCE,
    SUBJECT,
    TARGET,
    WEIGHT,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from itk_patterns.models import PatternConfig


def edge_frame(
    attribute_sets: Iterable[Sequence[str]],
    all_attributes: Sequence[str],
    min_edge_weight: float,
    missing_edge_prop: float,
) -> pl.DataFrame:
    """Weighted edges between every pair of ``all_attributes``.

    Observed pairs (attributes that co-occur in at least one set) have their
    co-occurrence count scaled linearly onto ``[min_edge_weight, 1]``; when
    every observed pair has the same count they all weigh ``1``. Pairs never
    observed together weigh ``missing_edge_prop * min_edge_weight``. Edges are
    returned with ``source < target`` and sorted by weight descending, then
    source, then target.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for attributes in attribute_sets:
        counts.update(
            (a, b) if a < b else (b, a)
            for a, b in combinations(sorted(set(attributes)), 2)
        )
    sources: list[str] = []
    targets: list[str] = []
    weights: list[float] = []
    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        span = max_count - min_count
        for (a, b), count in counts.items():
            scaled = (count - min_count) / span if span else 1.0
            sources.append(a)
            targets.append(b)
            weights.append(scaled * (1 - min_edge_weight) + min_edge_weight)
    missing_weight = missing_edge_prop * min_edge_weight
    ordered = sorted(set(all_attributes))
    for a, b in combinations(ordered, 2):
        if (a, b) not in counts:
            sources.append(a)
            targets.append(b)
            weights.append(missing_weight)
    frame = pl.DataFrame(
        {SOURCE: sources, TARGET: targets, WEIGHT: weights}, schema=EDGE_SCHEMA
    )
    return frame.sort([WEIGHT, SOURCE, TARGET], descending=[True, False, False])


def build_graph(edges: pl.DataFrame, nodes: Sequence[str] = ()) -> nx.Graph:
    """An undirected weighted graph with ``nodes`` and every row of ``edges``."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_weighted_edges_from(edges.select([SOURCE, TARGET, WEIGHT]).rows())
    return graph


def period_attribute_sets(links: pl.DataFrame, period: str) -> list[list[str]]:
    """The attributes of each subject observed in ``period``, sorted by subject."""
    grouped = (
        links.filter(pl.col(PERIOD) == period)
        .group_by(SUBJECT)
        .agg(pl.col(FULL_ATTRIBUTE).sort())
        .sort(SUBJECT)
    )
    return grouped.get_column(FULL_ATTRIBUTE).to_list()


def period_graphs(links: pl.DataFrame, config: PatternConfig) -> dict[str, nx.Graph]:
    """One graph per period over every attribute seen in any period.

    Each graph records its period as ``graph.graph["period"]``.
    """
    attributes = sorted(links.get_column(FULL_ATTRIBUTE).unique().to_list())
    graphs: dict[str, nx.Graph] = {}
    for period in sorted(links.get_column(PERIOD).unique().to_list()):
        edges = edge_frame(
            period_attribute_sets(links, period),
            attributes,
            config.min_edge_weight,
            config.missing_edge_prop,
        )
        graph = build_graph(edges, attributes)
        graph.graph["period"] = period
        graphs[period] = graph
    return graphs


__all__ = ["build_graph", "edge_frame", "period_attribute_sets", "period_graphs"]
