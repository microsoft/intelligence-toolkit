# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict
from typing import Any

import networkx as nx
import polars as pl
from graspologic.partition import hierarchical_leiden

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.risk_networks.config import DEFAULT_MAX_ATTRIBUTE_DEGREE, ENTITY_LABEL


# ruff: noqa
def trim_nodeset(
    graph: nx.Graph,
    additional_trimmed_attributes: list[str] | None = None,
    max_attribute_degree: int = DEFAULT_MAX_ATTRIBUTE_DEGREE,
) -> tuple[set, set[Any | str]]:
    if additional_trimmed_attributes is None:
        additional_trimmed_attributes = []

    trimmed_degrees = set()
    for node, degree in graph.degree():
        if not node.startswith(ENTITY_LABEL) and degree > max_attribute_degree:
            trimmed_degrees.add((node, degree))

    trimmed_nodes = {t[0] for t in trimmed_degrees}.union(additional_trimmed_attributes)
    return trimmed_degrees, trimmed_nodes


def get_entity_neighbors(overall_graph, inferred_links, trimmed_nodeset, node) -> list:
    if not len(overall_graph.nodes()):
        return []

    if node not in overall_graph.nodes():
        raise ValueError(f"Node {node} not in graph")

    neighbors = set(overall_graph.neighbors(node))
    if inferred_links:
        neighbors = neighbors.union(inferred_links[node])

    neighbors = neighbors.difference(trimmed_nodeset)

    return [neighbor for neighbor in sorted(neighbors) if neighbor != node]


def neighbor_is_valid(
    neighbor_node, supporting_attribute_types, trimmed_nodeset
) -> bool:
    if not neighbor_node:
        return False
    node_name = neighbor_node.split(ATTRIBUTE_VALUE_SEPARATOR)[0]
    is_not_supported = node_name not in supporting_attribute_types
    is_trimmed = neighbor_node in trimmed_nodeset
    return is_not_supported and not is_trimmed


def project_entity_graph(
    overall_graph: nx.Graph,
    trimmed_nodeset: set,
    inferred_links: dict[set],
    supporting_attribute_types: list[str],
) -> nx.Graph:
    P = nx.Graph()
    entity_nodes = [
        node for node in overall_graph.nodes() if node.startswith(ENTITY_LABEL)
    ]

    for node in entity_nodes:
        neighbors = get_entity_neighbors(
            overall_graph, inferred_links, trimmed_nodeset, node
        )

        for ent_neighbor in neighbors:
            if ent_neighbor.startswith(ENTITY_LABEL):
                P.add_edge(node, ent_neighbor)
            elif neighbor_is_valid(
                ent_neighbor, supporting_attribute_types, trimmed_nodeset
            ):
                att_neighbors = set(overall_graph.neighbors(ent_neighbor))
                if ent_neighbor in inferred_links:
                    att_neighbors = att_neighbors.union(inferred_links[ent_neighbor])
                for att_neighbor in att_neighbors:
                    if neighbor_is_valid(
                        att_neighbor, supporting_attribute_types, trimmed_nodeset
                    ):
                        if att_neighbor.startswith(ENTITY_LABEL):
                            if node != att_neighbor:
                                P.add_edge(node, att_neighbor)
                        else:  # fuzzy att link
                            fuzzy_att_neighbors = set(
                                overall_graph.neighbors(att_neighbor)
                            )
                            if att_neighbor in inferred_links:
                                fuzzy_att_neighbors = fuzzy_att_neighbors.union(
                                    inferred_links[att_neighbor]
                                )
                            for fuzzy_att_neighbor in fuzzy_att_neighbors:
                                if neighbor_is_valid(
                                    fuzzy_att_neighbor,
                                    supporting_attribute_types,
                                    trimmed_nodeset,
                                ):
                                    if fuzzy_att_neighbor.startswith(ENTITY_LABEL):
                                        if node != fuzzy_att_neighbor:
                                            P.add_edge(
                                                node,
                                                fuzzy_att_neighbor,
                                            )
    return P


def get_subgraph(
    entity_graph: nx.Graph,
    nodes: list[str | int],
    random_seed: int = 42,
    max_network_entities: int = 10,
) -> tuple[list, dict]:
    entity_to_community = {}
    community_nodes = []

    if not nodes or not entity_graph.nodes():
        return community_nodes, entity_to_community

    S = nx.subgraph(entity_graph, nodes)

    node_to_network = hierarchical_leiden(
        S,
        resolution=1.0,
        random_seed=random_seed,
        max_cluster_size=max_network_entities,
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
    max_network_entities: int = 50,
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
        if len(nodes) > max_network_entities:
            community_nodes_sequence, entity_to_community = get_subgraph(
                entity_graph, nodes, max_network_entities
            )
            community_nodes.extend(community_nodes_sequence)
            entity_to_community_ix.update(entity_to_community)
        else:
            community_nodes.append(nodes)
            for node in nodes:
                entity_to_community_ix[node] = len(community_nodes) - 1
    return community_nodes, entity_to_community_ix


def build_networks(
    main_graph: nx.Graph,
    trimmed_nodes: set,
    inferred_links: set | None = None,
    supporting_attribute_types: list[str] | None = None,
    max_network_entities: int = 20,
) -> tuple[list, dict]:
    P = project_entity_graph(
        main_graph, trimmed_nodes, inferred_links, supporting_attribute_types
    )

    (
        community_nodes,
        entity_to_community,
    ) = get_community_nodes(
        P,
        max_network_entities,
    )

    return community_nodes, entity_to_community


def get_integrated_flags(
    integrated_flags: pl.DataFrame, entities: list[str]
) -> tuple[Any, int, float, int]:
    if integrated_flags.is_empty():
        return 0, 0, 0, 0

    flags_df = integrated_flags.filter(pl.col("qualified_entity").is_in(entities))
    community_flags = flags_df.get_column("count").sum()
    flagged = flags_df.filter(pl.col("count") > 0).height
    unflagged = len(entities) - flagged
    flagged_per_unflagged = flagged / unflagged if unflagged > 0 else 0
    flagged_per_unflagged = round(flagged_per_unflagged, 2)

    flags_per_entity = round(
        community_flags / len(entities) if len(entities) > 0 else 0, 2
    )
    return community_flags, flagged, flagged_per_unflagged, flags_per_entity


def build_entity_records(
    community_nodes: list[str], integrated_flags: pl.DataFrame | None = None
) -> list[tuple[str, int, int, int, Any, int, float, float]]:
    if integrated_flags is None:
        integrated_flags = pl.DataFrame()

    entity_records = []
    for ix, entities in enumerate(community_nodes):
        (community_flags, flagged, flagged_per_unflagged, flags_per_entity) = (
            get_integrated_flags(integrated_flags, entities)
        )

        for n in entities:
            flags = 0
            if not integrated_flags.is_empty():
                flags = integrated_flags.filter(pl.col("qualified_entity") == n)[
                    "count"
                ].sum()

            entity_records.append(
                (
                    n.split(ATTRIBUTE_VALUE_SEPARATOR)[1],
                    flags,
                    ix,
                    len(entities),
                    community_flags,
                    flagged,
                    flags_per_entity,
                    flagged_per_unflagged,
                )
            )

    return entity_records
