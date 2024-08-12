# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import networkx as nx
from graspologic.partition import hierarchical_leiden


def get_subgraph(
    entity_graph: nx.Graph,
    nodes: list[str | int],
    random_seed: int = 42,
    max_cluster_size: int = 10,
) -> tuple[list, dict]:
    entity_to_community = {}
    community_nodes = []

    if not nodes or not entity_graph.nodes():
        return community_nodes, entity_to_community

    S = nx.subgraph(entity_graph, nodes)

    node_to_network = hierarchical_leiden(
        S, resolution=1.0, random_seed=random_seed, max_cluster_size=max_cluster_size
    ).final_level_hierarchical_clustering()

    network_to_nodes = defaultdict(set)
    for node, network in node_to_network.items():
        network_to_nodes[network].add(node)

    networks = [list(nodes) for nodes in network_to_nodes.values()]
    for network in networks:
        community_nodes.append(network)
        for node in network:
            entity_to_community[node] = len(community_nodes) - 1

    return community_nodes, entity_to_community


def get_community_nodes(
    entity_graph: nx.Graph,
    max_network_size: int = 50,
) -> tuple[list, dict]:
    # get set of connected nodes list
    sorted_components = sorted(
        nx.components.connected_components(entity_graph),
        key=lambda x: len(x),
        reverse=True,
    )

    components_sequence = range(len(sorted_components))
    component_to_nodes = dict(zip(components_sequence, sorted_components, strict=False))

    entity_to_community_ix = {}
    community_nodes = []
    for sequence in components_sequence:
        nodes = component_to_nodes[sequence]
        if len(nodes) > max_network_size:
            community_nodes_sequence, entity_to_community = get_subgraph(
                entity_graph, nodes, max_network_size
            )
            community_nodes.extend(community_nodes_sequence)
            entity_to_community_ix.update(entity_to_community)
        else:
            community_nodes.append(nodes)
            for node in nodes:
                entity_to_community_ix[node] = len(community_nodes) - 1
    return community_nodes, entity_to_community_ix
