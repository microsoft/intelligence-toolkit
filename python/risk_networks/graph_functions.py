# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from collections import defaultdict
from typing import Any

import networkx as nx

from . import config


def _merge_condition(x, y) -> bool:
    """
    Merge condition function for merging nodes in the graph.
    """
    x_parts = set(x.split(config.list_sep))
    y_parts = set(y.split(config.list_sep))
    return any(
        x_part.split(config.att_val_sep)[i] == y_part.split(config.att_val_sep)[i]
        for i in range(2)
        for x_part in x_parts
        for y_part in y_parts
    )


def _merge_node_list(G, merge_list) -> nx.Graph:  # noqa: N803
    G1 = G.copy()
    merged_node = config.list_sep.join(sorted(merge_list))
    merged_type = config.list_sep.join(sorted([G.nodes[n]["type"] for n in merge_list]))
    merged_risk = max(G.nodes[n]["flags"] for n in merge_list)
    G1.add_node(merged_node, type=merged_type, flags=merged_risk)
    for n in merge_list:
        for nn in G.neighbors(n):
            if nn not in merge_list:
                G1.add_edge(merged_node, nn)
        G1.remove_node(n)
    return G1


def _merge_nodes(G, merge_condition=_merge_condition) -> nx.Graph:  # noqa: N803
    nodes = list(G.nodes())  # may change during iteration
    for node in nodes:
        if node not in G.nodes():
            continue
        neighbours = list(G.neighbors(node))
        merge_list = [node]
        for n in neighbours:
            if n not in G.nodes():
                continue
            if merge_condition(node, n):
                merge_list.append(n)
        if len(merge_list) > 1:
            G = _merge_node_list(G, merge_list)

    return G


def simplify_graph(C) -> nx.Graph:  # noqa: N803
    S = C.copy()
    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    S = _merge_nodes(S)

    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    return S


def build_undirected_graph(
    network_attribute_links=[],  # noqa
    network_entity_links=[],  # noqa
) -> nx.Graph:
    G = nx.Graph()
    value_to_atts = defaultdict(set)
    for link_list in network_attribute_links:
        for link in link_list:
            n1 = f"{config.entity_label}{config.att_val_sep}{link[0]}"
            n2 = f"{link[1]}{config.att_val_sep}{link[2]}"
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            G.add_edge(edge[0], edge[1], type=link[1])
            G.add_node(n1, type=config.entity_label)
            G.add_node(n2, type=link[1])
            value_to_atts[link[2]].add(n2)

    for link_list in network_entity_links:
        n1 = f"{config.entity_label}{config.att_val_sep}{link_list[0]}"
        n2 = f"{config.entity_label}{config.att_val_sep}{link_list[2]}"
        edge = (n1, n2) if n1 < n2 else (n2, n1)
        G.add_edge(edge[0], edge[1], type=link_list[1])
        G.add_node(n1, type=config.entity_label)
        G.add_node(n2, type=config.entity_label)

    for atts in value_to_atts.values():
        att_list = list(atts)
        for i, att1 in enumerate(att_list):
            for att2 in att_list[i + 1 :]:
                edge = (att1, att2) if att1 < att2 else (att2, att1)
                G.add_edge(edge[0], edge[1], type="equality")
    return G  # network_overall_graph


def build_network_from_entities(
    G,  # noqa: N803
    nodes,
    network_trimmed_attributes,
    network_entity_to_community_ix,
    network_inferred_links,
    network_integrated_flags,
) -> tuple[nx.Graph, Any]:
    N = nx.Graph()
    trimmed_nodeset = network_trimmed_attributes["Attribute"].unique().tolist()
    for node in nodes:
        n_c = (
            str(network_entity_to_community_ix[node])
            if node in network_entity_to_community_ix
            else ""
        )
        N.add_node(node, type=config.entity_label, network=n_c, flags=0)
        ent_neighbors = set(G.neighbors(node)).union(network_inferred_links[node])
        for ent_neighbor in ent_neighbors:
            if ent_neighbor in trimmed_nodeset:
                continue

            if ent_neighbor.startswith(config.entity_label) and node != ent_neighbor:
                en_c = network_entity_to_community_ix.get(ent_neighbor, "")
                N.add_node(ent_neighbor, type=config.entity_label, network=en_c)
                N.add_edge(node, ent_neighbor)
            else:  # att
                N.add_node(
                    ent_neighbor,
                    type=ent_neighbor.split(config.att_val_sep)[0],
                    flags=0,
                )
                N.add_edge(node, ent_neighbor)
                att_neighbors = set(G.neighbors(ent_neighbor)).union(
                    network_inferred_links[ent_neighbor]
                )
                for att_neighbor in att_neighbors:
                    if att_neighbor in trimmed_nodeset or not att_neighbor.startswith(
                        config.entity_label
                    ):
                        continue

                    N.add_node(
                        att_neighbor,
                        type=att_neighbor.split(config.att_val_sep)[0],
                        flags=0,
                    )
                    fuzzy_att_neighbors = set(G.neighbors(att_neighbor)).union(
                        network_inferred_links[att_neighbor]
                    )
                    for fuzzy_att_neighbor in fuzzy_att_neighbors:
                        if (
                            fuzzy_att_neighbor in trimmed_nodeset
                            or fuzzy_att_neighbor.startswith(config.entity_label)
                        ):
                            continue
                        N.add_node(
                            fuzzy_att_neighbor,
                            type=fuzzy_att_neighbor.split(config.att_val_sep)[0],
                            flags=0,
                        )
                        N.add_edge(att_neighbor, fuzzy_att_neighbor)
    if len(network_integrated_flags) > 0:
        fdf = network_integrated_flags
        fdf = fdf[fdf["count"] > 0]
        flagged_nodes = fdf["qualified_entity"].unique().tolist()
        for node in flagged_nodes:
            if node in N.nodes():
                N.nodes[node]["flags"] = fdf.loc[
                    fdf["qualified_entity"] == node, "count"
                ].sum()
    return N, network_integrated_flags  # change sv???
