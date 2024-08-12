# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from typing import Any

import networkx as nx
import pandas as pd
from networkx import Graph

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.risk_networks import config


# ruff: noqa
def trim_nodeset(
    graph, additional_trimmed_attributes: list[str], max_attribute_degree: int
) -> tuple[set, set[Any | str]]:
    trimmed_degrees = set()
    for node, degree in graph.degree():
        if not node.startswith(config.entity_label) and degree > max_attribute_degree:
            trimmed_degrees.add((node, degree))

    # network_trimmed_attributes = (  # sv.network_trimmed_attributes.value return
    #     pd.DataFrame(trim, columns=["Attribute", "Linked Entities"])
    #     .sort_values("Linked Entities", ascending=False)
    #     .reset_index(drop=True)
    # )
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
    inferred_links: dict,
    supporting_attribute_types: list[str],
) -> Graph:
    P = nx.Graph()
    entity_nodes = [
        node for node in overall_graph.nodes() if node.startswith(config.entity_label)
    ]

    for node in entity_nodes:
        neighbors = get_entity_neighbors(
            overall_graph, inferred_links, trimmed_nodeset, node
        )

        for ent_neighbor in neighbors:
            if ent_neighbor.startswith(config.entity_label):
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
                        if att_neighbor.startswith(config.entity_label):
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
                                    if fuzzy_att_neighbor.startswith(
                                        config.entity_label
                                    ):
                                        if node != fuzzy_att_neighbor:
                                            P.add_edge(
                                                node,
                                                fuzzy_att_neighbor,
                                            )
    return P
