# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Split the projected entity graph into networks of bounded size."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

import networkx as nx

from itk_networks.models import Network

if TYPE_CHECKING:
    from itk_networks.models import NetworkConfig


class CommunityDetector(Protocol):
    """Assigns every node of ``graph`` to a cluster, aiming for ``max_size`` nodes.

    ``max_size`` is a target, not a guarantee: graspologic's hierarchical Leiden
    leaves a cluster whole when it cannot split it (a clique, for instance).
    """

    def detect(self, graph: nx.Graph, max_size: int, seed: int) -> dict[str, int]:
        """Return ``node -> cluster id``; cluster ids need only be distinct."""
        ...


def default_detector() -> CommunityDetector:
    """The graspologic Leiden adapter, if the ``leiden`` extra is installed."""
    try:
        from itk_networks.leiden import GraspologicLeiden
    except ImportError as e:  # pragma: no cover - depends on the environment
        msg = (
            "A component is larger than max_network_entities and no "
            "CommunityDetector was given; install itk-networks[leiden] or pass one"
        )
        raise ImportError(msg) from e
    return GraspologicLeiden()


def _ordered(groups: list[set[str] | frozenset[str]]) -> list[tuple[str, ...]]:
    """Largest first, ties broken by smallest member, members sorted."""
    return [
        tuple(sorted(group))
        for group in sorted(groups, key=lambda g: (-len(g), min(g)))
    ]


def split_component(
    projected: nx.Graph,
    nodes: tuple[str, ...],
    max_size: int,
    seed: int,
    detector: CommunityDetector,
) -> list[tuple[str, ...]]:
    """Clusters of one oversized component, ordered like top-level components.

    The subgraph is rebuilt with nodes and edges in sorted order: Leiden's result
    depends on insertion order, and the projection's order comes from set
    iteration, so without this the partition changes from process to process.
    """
    subgraph = nx.Graph()
    subgraph.add_nodes_from(sorted(nodes))
    subgraph.add_edges_from(
        sorted(tuple(sorted(edge)) for edge in projected.subgraph(nodes).edges())
    )
    assignment = detector.detect(subgraph, max_size, seed)
    missing = sorted(set(nodes) - set(assignment))
    if missing:
        msg = f"Detector left {len(missing)} node(s) unassigned"
        raise ValueError(msg)
    clusters: defaultdict[int, set[str]] = defaultdict(set)
    for node in nodes:
        clusters[assignment[node]].add(node)
    return _ordered(list(clusters.values()))


def find_networks(
    projected: nx.Graph,
    config: NetworkConfig,
    detector: CommunityDetector | None = None,
) -> tuple[list[Network], dict[str, int]]:
    """Connected components, largest first; oversized ones split by the detector.

    Network ids are positions in the returned list. The detector is only
    resolved when a component actually needs splitting.
    """
    networks: list[Network] = []
    membership: dict[str, int] = {}
    for component in _ordered(list(nx.connected_components(projected))):
        if len(component) > config.max_network_entities:
            detector = detector or default_detector()
            parts = split_component(
                projected,
                component,
                config.max_network_entities,
                config.leiden_seed,
                detector,
            )
        else:
            parts = [component]
        for part in parts:
            network = Network(id=len(networks), entities=part)
            networks.append(network)
            for node in part:
                membership[node] = network.id
    return networks, membership
