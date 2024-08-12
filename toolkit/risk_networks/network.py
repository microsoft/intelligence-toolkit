# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from typing import Any

import networkx as nx
import polars as pl

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.risk_networks import config


# I don't have the flags yet
def build_integrated_flags(
    graph: nx.Graph, df_integrated_flags: pl.DataFrame
) -> nx.Graph:
    df_integrated_flags = df_integrated_flags[df_integrated_flags["count"] > 0]
    flagged_nodes = df_integrated_flags["qualified_entity"].unique().tolist()
    # get only nodes in graph.nodes()
    # flagged_nodes = [node for node in flagged_nodes if node in graph.nodes()]
    for node in flagged_nodes:
        if node in graph.nodes():
            graph.nodes[node]["flags"] = df_integrated_flags[
                df_integrated_flags["qualified_entity"] == node
            ]["count"].sum()
    return graph


def build_fuzzy_neighbors(
    graph: nx.Graph,
    network_graph: nx.Graph,
    att_neighbor: str | int,
    trimmed_nodeset: set,
    inferred_links: dict,
) -> nx.Graph:
    if not len(graph.nodes()):
        return nx.Graph()

    if att_neighbor not in graph.nodes():
        msg = f"Node {att_neighbor} not in graph"
        raise ValueError(msg)

    fuzzy_att_neighbors = set(graph.neighbors(att_neighbor))
    if att_neighbor in inferred_links:
        fuzzy_att_neighbors = fuzzy_att_neighbors.union(inferred_links[att_neighbor])

    fuzzy_att_neighbors_not_trimmed = [
        fuzzy_att_neighbor
        for fuzzy_att_neighbor in fuzzy_att_neighbors
        if fuzzy_att_neighbor not in trimmed_nodeset
        and not fuzzy_att_neighbor.startswith(config.entity_label)
    ]
    for fuzzy_att_neighbor in fuzzy_att_neighbors_not_trimmed:
        network_graph.add_node(
            fuzzy_att_neighbor,
            type=fuzzy_att_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
            flags=0,
        )
        network_graph.add_edge(att_neighbor, fuzzy_att_neighbor)
    return network_graph


def build_network_from_entities(
    graph, entity_to_community, integrated_flags, trimmed_attributes, inferred_links
) -> nx.Graph:
    network_graph = nx.Graph()
    nodes = graph.nodes()
    # additional_trimmed_nodeset = set(sv.network_additional_trimmed_attributes.value)
    trimmed_nodeset = trimmed_attributes["Attribute"].unique().tolist()
    # trimmed_nodeset.extend(additional_trimmed_nodeset)
    for node in nodes:
        n_c = str(entity_to_community[node]) if node in entity_to_community else ""
        network_graph.add_node(node, type=config.entity_label, network=n_c, flags=0)
        ent_neighbors = set(graph.neighbors(node))
        if node in inferred_links:
            ent_neighbors = ent_neighbors.union(inferred_links[node])

        ent_neighbors_not_trimmed = [
            ent_neighbor
            for ent_neighbor in ent_neighbors
            if ent_neighbor not in trimmed_nodeset
        ]

        for ent_neighbor in ent_neighbors_not_trimmed:
            if ent_neighbor.startswith(config.entity_label):
                if node != ent_neighbor:
                    en_c = entity_to_community.get(ent_neighbor, "")
                    network_graph.add_node(
                        ent_neighbor, type=config.entity_label, network=en_c
                    )
                    network_graph.add_edge(node, ent_neighbor)
            else:
                network_graph.add_node(
                    ent_neighbor,
                    type=ent_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                    flags=0,
                )
                network_graph.add_edge(node, ent_neighbor)
                att_neighbors = set(graph.neighbors(ent_neighbor)).union(
                    inferred_links[ent_neighbor]
                )
                att_neighbors_not_trimmed = [
                    att_neighbor
                    for att_neighbor in att_neighbors
                    if att_neighbor not in trimmed_nodeset
                    and not att_neighbor.startswith(config.entity_label)
                ]
                for att_neighbor in att_neighbors_not_trimmed:
                    network_graph.add_node(
                        att_neighbor,
                        type=att_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                        flags=0,
                    )
                    network_graph = build_fuzzy_neighbors(
                        graph,
                        network_graph,
                        att_neighbor,
                        trimmed_nodeset,
                        inferred_links,
                    )

    if len(integrated_flags) > 0:
        network_graph = build_integrated_flags(network_graph, integrated_flags)
    return network_graph


def get_integrated_flags(
    integrated_flags: pl.DataFrame, entities: list[str]
) -> tuple[Any, int, float, int]:
    if integrated_flags.is_empty():
        return 0, 0, 0, 0
    # Assuming integrated_flags is a pl.DataFrame and entities is a list or pl.Series
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


def generate_final_df(
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
