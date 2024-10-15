# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import streamlit as st
from seaborn import color_palette
from streamlit_agraph import Config, Edge, Node, agraph

from toolkit.helpers.progress_batch_callback import ProgressBatchCallback


def create_progress_callback(template: str):
    pb = st.progress(0, "Preparing...")

    def on_change(current, total):
        pb.progress(
            (current) / total if (current) / total < 100 else 100,
            text=template.format(current, total),
        )

    callback = ProgressBatchCallback()
    callback.on_batch_change = on_change
    return pb, callback


def get_concept_graph(
    placeholder, G, hierarchical_communities, width, height, key
):
    """
    Implements the concept graph visualization
    """
    nodes = []
    edges = []
    max_degree = max([G.degree(node) for node in G.nodes()])
    concept_to_community = hierarchical_communities.final_level_hierarchical_clustering()
    community_to_concepts = defaultdict(set)
    for concept, community in concept_to_community.items():
        community_to_concepts[community].add(concept)

    num_communities = len(community_to_concepts.keys())
    community_colors = color_palette("husl", num_communities)
    sorted_communities = sorted(
        community_to_concepts.keys(),
        key=lambda x: len(community_to_concepts[x]),
        reverse=True,
    )
    community_to_color = dict(zip(sorted_communities, community_colors))
    for node in G.nodes():
        if node == "dummynode":
            continue
        degree = G.degree(node)
        size = 5 + 20 * degree / max_degree
        vadjust = -size * 2 - 3
        community = concept_to_community[node] if node in concept_to_community else -1
        color = (
            community_to_color[community]
            if community in community_to_color
            else (0.75, 0.75, 0.75)
        )
        color = "#%02x%02x%02x" % tuple([int(255 * x) for x in color])
        nodes.append(
            Node(
                title=node,
                id=node,
                label=node,
                size=size,
                color=color,
                shape="dot",
                timestep=0.001,
                font={"vadjust": vadjust, "size": size},
            )
        )

    for u, v, d in G.edges(data=True):
        if u == "dummynode" or v == "dummynode":
            continue
        edges.append(Edge(source=u, target=v, color="lightgray"))

    config = Config(
        width=width,
        height=height,
        directed=False,
        physics=True,
        hierarchical=False,
        key=key,
        linkLength=100,
        timestep=0.1,
    )
    with placeholder:
        return_value = agraph(nodes=nodes, edges=edges, config=config)
    return return_value