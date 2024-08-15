# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import colorsys
from collections import defaultdict

# ruff: noqa
import networkx as nx
import pandas as pd
import polars as pl
import streamlit as st
from streamlit_agraph import Config, Edge, Node
from util.openai_wrapper import UIOpenAIConfiguration
from util.session_variables import SessionVariables

from toolkit.AI.embedder import Embedder
from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.risk_networks import config

sv_home = SessionVariables("home")


def embedder():
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        return Embedder(
            ai_configuration, config.cache_dir, sv_home.local_embeddings.value
        )
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

            parts = [
                p.split(ATTRIBUTE_VALUE_SEPARATOR) for p in node.split(config.list_sep)
            ]
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
