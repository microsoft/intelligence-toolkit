# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import colorsys
from typing import Any

import networkx as nx
import polars as pl

from toolkit.detect_entity_networks.config import ENTITY_LABEL, LIST_SEPARATOR
from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR


def _integrate_flags(graph: nx.Graph, df_integrated_flags: pl.DataFrame) -> nx.Graph:
    if not graph.nodes() or df_integrated_flags.is_empty():
        return nx.Graph()

    df_integrated_flags = df_integrated_flags.filter(pl.col("count") > 0)
    flagged_nodes = (
        df_integrated_flags.select("qualified_entity").unique().to_series().to_list()
    )

    flagged_nodes = [node for node in flagged_nodes if node in graph.nodes()]
    for node in flagged_nodes:
        graph.nodes[node]["flags"] = df_integrated_flags.filter(
            pl.col("qualified_entity") == node
        )["count"].sum()
    return graph


def _build_fuzzy_neighbors(
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
        and not fuzzy_att_neighbor.startswith(ENTITY_LABEL)
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
    graph,
    entity_to_community,
    integrated_flags: pl.DataFrame | None = None,
    trimmed_attributes: list[tuple[str, int]] | None = None,
    inferred_links: Any | None = None,
    selected_nodes: list[str] | None = None,
) -> nx.Graph:
    network_graph = nx.Graph()
    nodes = selected_nodes
    # additional_trimmed_nodeset = set(sv.network_additional_trimmed_attributes.value)
    # trimmed_nodeset = trimmed_attributes["Attribute"].unique().tolist()
    trimmed_nodeset = {t[0] for t in trimmed_attributes}

    if inferred_links is None:
        inferred_links = {}

    if integrated_flags is None:
        integrated_flags = pl.DataFrame()

    # trimmed_nodeset.extend(additional_trimmed_nodeset)
    for node in nodes:
        n_c = str(entity_to_community[node]) if node in entity_to_community else ""
        network_graph.add_node(node, type=ENTITY_LABEL, network=n_c, flags=0)
        ent_neighbors = set(graph.neighbors(node))
        if node in inferred_links:
            ent_neighbors = ent_neighbors.union(inferred_links[node])

        ent_neighbors_not_trimmed = [
            ent_neighbor
            for ent_neighbor in ent_neighbors
            if ent_neighbor not in trimmed_nodeset
        ]

        for ent_neighbor in ent_neighbors_not_trimmed:
            if ent_neighbor.startswith(ENTITY_LABEL):
                if node != ent_neighbor:
                    en_c = entity_to_community.get(ent_neighbor, "")
                    network_graph.add_node(
                        ent_neighbor, type=ENTITY_LABEL, network=en_c
                    )
                    network_graph.add_edge(node, ent_neighbor)
            else:
                network_graph.add_node(
                    ent_neighbor,
                    type=ent_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                    flags=0,
                )
                network_graph.add_edge(node, ent_neighbor)
                att_neighbors = set(graph.neighbors(ent_neighbor))
                if ent_neighbor in inferred_links:
                    att_neighbors = att_neighbors.union(inferred_links[ent_neighbor])
                att_neighbors_not_trimmed = [
                    att_neighbor
                    for att_neighbor in att_neighbors
                    if att_neighbor not in trimmed_nodeset
                    and not att_neighbor.startswith(ENTITY_LABEL)
                ]
                for att_neighbor in att_neighbors_not_trimmed:
                    network_graph.add_node(
                        att_neighbor,
                        type=att_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                        flags=0,
                    )
                    network_graph = _build_fuzzy_neighbors(
                        graph,
                        network_graph,
                        att_neighbor,
                        trimmed_nodeset,
                        inferred_links,
                    )

    if len(integrated_flags) > 0:
        network_graph = _integrate_flags(network_graph, integrated_flags)
    return network_graph


def _merge_condition(x: str, y: str) -> bool:
    """
    Merge condition function for merging nodes in the graph.
    """
    x_parts = set(x.split(LIST_SEPARATOR))
    y_parts = set(y.split(LIST_SEPARATOR))
    return any(
        x_part.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
        == y_part.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
        for i in range(2)
        for x_part in x_parts
        for y_part in y_parts
    )


def _merge_node_list(graph: nx.Graph, merge_list: list[str]) -> nx.Graph:
    graph = graph.copy()
    merged_node = LIST_SEPARATOR.join(sorted(merge_list))
    merged_type = LIST_SEPARATOR.join(
        sorted([graph.nodes[n]["type"] for n in merge_list])
    )
    merged_risk = max(graph.nodes[n]["flags"] if "flags" in graph.nodes[n] else 0 for n in merge_list)
    graph.add_node(merged_node, type=merged_type, flags=merged_risk)
    for n in merge_list:
        for nn in graph.neighbors(n):
            if nn not in merge_list:
                graph.add_edge(merged_node, nn)
        graph.remove_node(n)
    return graph


def _merge_nodes(graph: nx.Graph, should_merge=_merge_condition) -> nx.Graph:
    nodes = list(graph.nodes())  # may change during iteration
    for node in nodes:
        if node not in graph.nodes():
            continue
        neighbours = list(graph.neighbors(node))
        merge_list = [node]
        for n in neighbours:
            if n not in graph.nodes():
                continue
            if should_merge(node, n):
                merge_list.append(n)
        if len(merge_list) > 1:
            graph = _merge_node_list(graph, merge_list)

    return graph


def simplify_entities_graph(entities_graph: nx.Graph) -> nx.Graph:
    # remove single degree attributes
    entities_graph = entities_graph.copy()
    for node in list(entities_graph.nodes()):
        if entities_graph.degree(node) < 2 and not node.startswith(ENTITY_LABEL):
            entities_graph.remove_node(node)

    entities_graph = _merge_nodes(entities_graph)

    # remove single degree attributes
    for node in list(entities_graph.nodes()):
        if entities_graph.degree(node) < 2 and not node.startswith(ENTITY_LABEL):
            entities_graph.remove_node(node)

    return entities_graph


def hsl_to_hex(hue: int, saturation: int, lightness: int) -> str:
    rgb = colorsys.hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
    return "#{:02x}{:02x}{:02x}".format(*tuple(int(c * 255) for c in rgb))


def get_type_color(node_type: str, is_flagged: bool, attribute_types: list[Any]) -> str:
    if is_flagged:
        hue = 0
        saturation = 70
        lightness = 80
    else:
        start = 230
        reserve = 35
        prop = attribute_types.index(node_type) / len(attribute_types)
        inc = prop * (360 - 2 * reserve)
        # avoid reds
        hue = (start + inc) % 360
        if hue < reserve:
            hue += 2 * reserve
        if hue > 360 - reserve:
            hue = (hue + 2 * reserve) % 360
        saturation = 70
        lightness = 80
    return str(hsl_to_hex(hue, saturation, lightness))


def get_entity_graph(
    network_entities_graph: nx.Graph, selected: str, attribute_types: list[str]
) -> tuple[list, list]:
    """
    Implements the entity graph visualization after network selection
    """
    node_names = set()
    nodes = []
    edges = []

    if not network_entities_graph.edges():
        return nodes, edges

    links_df = pl.DataFrame(
        list(network_entities_graph.edges()), schema=["source", "target"]
    )

    links_df.with_columns(
        [
            pl.col("target")
            .map_elements(lambda x: x.split(ATTRIBUTE_VALUE_SEPARATOR)[0])
            .alias("attribute")
        ]
    )

    all_nodes = set(links_df["source"]).union(set(links_df["target"]))
    for node in all_nodes:
        node_names.add(node)
        size = 20 if node == selected else 12 if node.startswith(ENTITY_LABEL) else 8
        vadjust = -size - 10

        parts = [p.split(ATTRIBUTE_VALUE_SEPARATOR) for p in node.split(LIST_SEPARATOR)]
        atts = [p[0] for p in parts]
        atts = list(dict.fromkeys(atts))

        flags = network_entities_graph.nodes[node].get("flags", 0)
        color = get_type_color(
            atts[0],
            flags > 0,
            attribute_types,
        )

        vals = [p[1] for p in parts if len(p) > 1]
        vals = list(dict.fromkeys(vals))
        label = "\n".join(vals) + "\n(" + LIST_SEPARATOR.join(atts) + ")"

        nodes.append(
            {
                "title": node + f"\nFlags: {flags}",
                "id": node,
                "label": label,
                "size": size,
                "color": color,
                "font": {"vadjust": vadjust, "size": 5},
            }
        )
    for row in list(network_entities_graph.edges()):
        source = row[0]
        target = row[1]
        edges.append(
            {
                "source": source,
                "target": target,
                "color": "mediumgray",
                "size": 1,
            }
        )
    return nodes, edges
