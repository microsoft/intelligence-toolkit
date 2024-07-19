# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import colorsys
from collections import defaultdict

# ruff: noqa
import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node
from util.openai_wrapper import UIOpenAIConfiguration

from python.AI.embedder import Embedder
from python.risk_networks import config


def embedder():
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        return Embedder(ai_configuration, config.cache_dir)
    except Exception as e:
        st.error(f"Error creating connection: {e}")
        st.stop()


def hsl_to_hex(h, s, l):
    rgb = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    return "#{:02x}{:02x}{:02x}".format(*tuple(int(c * 255) for c in rgb))


def get_entity_graph(G, selected, links_df, width, height, attribute_types):
    """
    Implements the entity graph visualization after network selection
    """
    node_names = set()
    nodes = []
    edges = []
    all_nodes = set(links_df["source"]).union(set(links_df["target"]))
    if len(all_nodes) > 0:
        max(G.nodes[node]["flags"] for node in all_nodes)
        for node in all_nodes:
            node_names.add(node)
            size = (
                20
                if node == selected
                else 12
                if node.startswith(config.entity_label)
                else 8
            )
            vadjust = -size - 10

            def get_type_color(node_type, is_flagged, attribute_types):
                if is_flagged:
                    h = 0
                    s = 70
                    l = 80
                else:
                    start = 230
                    reserve = 35
                    prop = attribute_types.index(node_type) / len(attribute_types)
                    inc = prop * (360 - 2 * reserve)
                    # avoid reds
                    h = (start + inc) % 360
                    if h < reserve:
                        h += 2 * reserve
                    if h > 360 - reserve:
                        h = (h + 2 * reserve) % 360
                    s = 70
                    l = 80
                return str(hsl_to_hex(h, s, l))

            parts = [p.split(config.att_val_sep) for p in node.split(config.list_sep)]
            atts = [p[0] for p in parts]
            # remove duplicate values while maintaining order
            atts = list(dict.fromkeys(atts))
            color = get_type_color(atts[0], G.nodes[node]["flags"] > 0, attribute_types)
            vals = [p[1] for p in parts if len(p) > 1]
            # remove duplicate values while maintaining order
            vals = list(dict.fromkeys(vals))
            G.nodes[node].get("network", "")
            label = "\n".join(vals) + "\n(" + config.list_sep.join(atts) + ")"
            d_risk = G.nodes[node]["flags"]

            nodes.append(
                Node(
                    title=node + f"\nFlags: {d_risk}",
                    id=node,
                    label=label,
                    size=size,
                    color=color,
                    font={
                        "vadjust": vadjust,
                        "size": 5,
                    },  # , 'size' : '8px', 'face': 'arial', 'color' : 'black'},
                )
            )
        for _i, row in links_df.iterrows():
            source = row["source"]
            target = row["target"]
            edges.append(Edge(source=source, target=target, color="mediumgray", size=1))
    g_config = Config(
        width=width, height=height, directed=False, physics=True, hierarchical=False
    )
    return nodes, edges, g_config  # type: ignore


def merge_nodes(G, can_merge_fn):
    nodes = list(G.nodes())  # may change during iteration
    for _ix, node in enumerate(nodes):
        if node not in G.nodes():
            continue
        neighbours = list(G.neighbors(node))
        merge_list = [node]
        for n in neighbours:
            if n not in G.nodes():
                continue
            if can_merge_fn(node, n):
                merge_list.append(n)
        if len(merge_list) > 1:
            G = merge_node_list(G, merge_list)

    return G


def merge_node_list(G, merge_list):
    G1 = G.copy()
    m = config.list_sep.join(sorted(merge_list))
    t = config.list_sep.join(sorted([G.nodes[n]["type"] for n in merge_list]))
    merged_risk = max(G.nodes[n]["flags"] for n in merge_list)
    G1.add_node(m, type=t)
    G1.nodes[m]["flags"] = merged_risk
    has_flags = False
    for n in merge_list:
        for nn in G.neighbors(n):
            if nn not in merge_list:
                G1.add_edge(m, nn)
        G1.remove_node(n)
        has_flags = has_flags or G.nodes[n]["flags"] > 0
    return G1


def merge_paths(in_paths):
    return in_paths


def simplify_graph(C):
    S = C.copy()
    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    S = merge_nodes(
        S,
        lambda x, y:  # merge if overlapping types or values
        len(
            {
                xv.split(config.att_val_sep)[0]
                for xv in sorted(x.split(config.list_sep))
            }.intersection({
                yv.split(config.att_val_sep)[0]
                for yv in sorted(y.split(config.list_sep))
            })
        )
        > 0
        or len(
            {
                xv.split(config.att_val_sep)[1]
                for xv in sorted(x.split(config.list_sep))
            }.intersection({
                yv.split(config.att_val_sep)[1]
                for yv in sorted(y.split(config.list_sep))
            })
        )
        > 0,
    )

    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    return S


def project_entity_graph(sv):
    # Remove high-degree attributes
    trim = [
        (n, d)
        for (n, d) in sv.network_overall_graph.value.degree()
        if not n.startswith(config.entity_label)
        and d > sv.network_max_attribute_degree.value
    ]
    trimmed_nodeset = {t[0] for t in trim}.union(
        sv.network_additional_trimmed_attributes.value
    )
    sv.network_trimmed_attributes.value = (
        pd.DataFrame(trim, columns=["Attribute", "Linked Entities"])
        .sort_values("Linked Entities", ascending=False)
        .reset_index(drop=True)
    )
    P = nx.Graph()
    sv.network_entity_graph.value = P
    for node in sv.network_overall_graph.value.nodes():
        if node.startswith(config.entity_label):
            ent_neighbors = set(sv.network_overall_graph.value.neighbors(node)).union(
                sv.network_inferred_links.value[node]
            )
            for ent_neighbor in ent_neighbors:
                if ent_neighbor not in trimmed_nodeset:
                    if ent_neighbor.startswith(config.entity_label):
                        if node != ent_neighbor:
                            P.add_edge(node, ent_neighbor)
                    else:  # att
                        if (
                            ent_neighbor.split(config.att_val_sep)[0]
                            not in sv.network_supporting_attribute_types.value
                        ):
                            att_neighbors = set(
                                sv.network_overall_graph.value.neighbors(ent_neighbor)
                            ).union(sv.network_inferred_links.value[ent_neighbor])
                            for att_neighbor in att_neighbors:
                                if (
                                    att_neighbor.split(config.att_val_sep)[0]
                                    not in sv.network_supporting_attribute_types.value
                                ):
                                    if att_neighbor not in trimmed_nodeset:
                                        if att_neighbor.startswith(config.entity_label):
                                            if node != att_neighbor:
                                                P.add_edge(node, att_neighbor)
                                        else:  # fuzzy att link
                                            fuzzy_att_neighbors = set(
                                                sv.network_overall_graph.value.neighbors(
                                                    att_neighbor
                                                )
                                            ).union(
                                                sv.network_inferred_links.value[
                                                    att_neighbor
                                                ]
                                            )
                                            for (
                                                fuzzy_att_neighbor
                                            ) in fuzzy_att_neighbors:
                                                if (
                                                    fuzzy_att_neighbor.split(
                                                        config.att_val_sep
                                                    )[0]
                                                    not in sv.network_supporting_attribute_types.value
                                                ):
                                                    if (
                                                        fuzzy_att_neighbor
                                                        not in trimmed_nodeset
                                                    ):
                                                        if fuzzy_att_neighbor.startswith(
                                                            config.entity_label
                                                        ):
                                                            if (
                                                                node
                                                                != fuzzy_att_neighbor
                                                            ):
                                                                P.add_edge(
                                                                    node,
                                                                    fuzzy_att_neighbor,
                                                                )
    return P


def build_undirected_graph(sv):
    G = nx.Graph()
    sv.network_overall_graph.value = G
    value_to_atts = defaultdict(set)
    for link_list in sv.network_attribute_links.value:
        for link in link_list:
            n1 = f"{config.entity_label}{config.att_val_sep}{link[0]}"
            n2 = f"{link[1]}{config.att_val_sep}{link[2]}"
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            G.add_edge(edge[0], edge[1], type=link[1])
            G.add_node(n1, type=config.entity_label)
            G.add_node(n2, type=link[1])
            value_to_atts[link[2]].add(n2)
    for link_list in sv.network_entity_links.value:
        for link in link_list:
            n1 = f"{config.entity_label}{config.att_val_sep}{link[0]}"
            n2 = f"{config.entity_label}{config.att_val_sep}{link[2]}"
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            G.add_edge(edge[0], edge[1], type=link[1])
            G.add_node(n1, type=config.entity_label)
            G.add_node(n2, type=config.entity_label)
    for atts in value_to_atts.values():
        att_list = list(atts)
        for i, att1 in enumerate(att_list):
            for att2 in att_list[i + 1 :]:
                edge = (att1, att2) if att1 < att2 else (att2, att1)
                G.add_edge(edge[0], edge[1], type="equality")
    return G


def build_integrated_flags(sv):
    sv.network_integrated_flags.value = pd.concat([
        pd.DataFrame(link_list, columns=["entity", "type", "flag", "count"])
        for link_list in sv.network_flag_links.value
    ])
    sv.network_integrated_flags.value = (
        sv.network_integrated_flags.value.groupby(["entity", "type", "flag"])
        .sum()
        .reset_index()
    )
    sv.network_integrated_flags.value["qualified_entity"] = (
        sv.network_integrated_flags.value[
            "entity"
        ].apply(lambda x: f"{config.entity_label}{config.att_val_sep}{x}")
    )
    overall_df = (
        sv.network_integrated_flags.value[["qualified_entity", "count"]]
        .groupby("qualified_entity")
        .sum()
        .reset_index()
    )
    sv.network_max_entity_flags.value = overall_df["count"].max()
    sv.network_mean_flagged_flags.value = round(
        overall_df[overall_df["count"] > 0]["count"].mean(), 2
    )


def build_network_from_entities(sv, G, nodes):
    N = nx.Graph()
    trimmed_nodeset = sv.network_trimmed_attributes.value["Attribute"].unique().tolist()
    for node in nodes:
        n_c = (
            str(sv.network_entity_to_community_ix.value[node])
            if node in sv.network_entity_to_community_ix.value
            else ""
        )
        N.add_node(node, type=config.entity_label, network=n_c, flags=0)
        ent_neighbors = set(G.neighbors(node)).union(
            sv.network_inferred_links.value[node]
        )
        for ent_neighbor in ent_neighbors:
            if ent_neighbor not in trimmed_nodeset:
                if ent_neighbor.startswith(config.entity_label):
                    if node != ent_neighbor:
                        en_c = sv.network_entity_to_community_ix.value.get(
                            ent_neighbor, ""
                        )
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
                        sv.network_inferred_links.value[ent_neighbor]
                    )
                    for att_neighbor in att_neighbors:
                        if att_neighbor not in trimmed_nodeset:
                            if not att_neighbor.startswith(config.entity_label):
                                N.add_node(
                                    att_neighbor,
                                    type=att_neighbor.split(config.att_val_sep)[0],
                                    flags=0,
                                )
                                fuzzy_att_neighbors = set(
                                    G.neighbors(att_neighbor)
                                ).union(sv.network_inferred_links.value[att_neighbor])
                                for fuzzy_att_neighbor in fuzzy_att_neighbors:
                                    if fuzzy_att_neighbor not in trimmed_nodeset:
                                        N.add_node(
                                            fuzzy_att_neighbor,
                                            type=fuzzy_att_neighbor.split(
                                                config.att_val_sep
                                            )[0],
                                            flags=0,
                                        )
                                        N.add_edge(att_neighbor, fuzzy_att_neighbor)
    if len(sv.network_integrated_flags.value) > 0:
        fdf = sv.network_integrated_flags.value
        fdf = fdf[fdf["count"] > 0]
        flagged_nodes = fdf["qualified_entity"].unique().tolist()
        for node in flagged_nodes:
            if node in N.nodes():
                N.nodes[node]["flags"] = fdf[fdf["qualified_entity"] == node][
                    "count"
                ].sum()
    return N
