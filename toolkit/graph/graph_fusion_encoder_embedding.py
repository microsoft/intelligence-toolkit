# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import networkx as nx
from collections import defaultdict

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.graph.graph_encoder_embed import GraphEncoderEmbed

def _get_edge_list(graph, node_list):
    """Generate a list of edges with weights for existing nodes."""
    node_to_ix = {node: i for i, node in enumerate(node_list)}
    return [
        [node_to_ix[s], node_to_ix[t], w]
        for s, t, w in graph.edges(data="weight")
        if s in node_list and t in node_list
    ]


def _generate_embeddings_for_period(graph, node_list, node_to_label, correlation, diaga, laplacian):
    """Generate embeddings for a single period."""
    edge_list = _get_edge_list(graph, node_list)
    num_nodes = len(node_list)
    labels = np.array([node_to_label[node] if node in node_to_label else -1 for node in node_list ]).reshape(
        (
            num_nodes,
            1,
        )
    )
    Z, _ = GraphEncoderEmbed().run(
        edge_list,
        labels,
        num_nodes,
        EdgeList=True,
        Laplacian=laplacian,
        DiagA=diaga,
        Correlation=correlation,
    )
    return Z.toarray()

def _cosine_distance(x, y):
    den = np.linalg.norm(x) * np.linalg.norm(y)
    dist = 1 - (np.dot(x, y) / den) if den > 0 else np.inf
    return dist

def generate_graph_fusion_encoder_embedding(period_to_graph, node_to_label, correlation, diaga, laplacian):
    """Generate embeddings for all periods and calculate centroids.
    All-time centroids are encoded as 'ALL' and prior centroids are encoded as '<'+period.
    """
    node_list = sorted(node_to_label.keys())
    node_to_period_to_pos = defaultdict(lambda: defaultdict(np.array))
    node_to_period_to_shift = defaultdict(lambda: defaultdict(np.array))
    for period, graph in period_to_graph.items():
        period_embedding = _generate_embeddings_for_period(
            graph, node_list, node_to_label, correlation, diaga, laplacian
        )
        for node_id in range(len(period_embedding)):
            node_to_period_to_pos[node_list[node_id]][period] = period_embedding[node_id]

    for node, period_to_pos in node_to_period_to_pos.items():
        all_positions = [pos for period, pos in period_to_pos.items()]
        centroid = np.mean(all_positions, axis=0)
        node_to_period_to_pos[node]['ALL'] = centroid
        sorted_periods = sorted(period_to_pos.keys())
        prior_positions = []
        for period in sorted_periods:
            if len(prior_positions) > 0:
                # Encodes prior centroid position and shift
                prior_centroid = np.mean(prior_positions, axis=0)
                node_to_period_to_pos[node]['<' + period] = prior_centroid
                node_to_period_to_shift[node]['<' + period] = _cosine_distance(period_to_pos[period], prior_centroid)
            prior_positions.append(period_to_pos[period])
        node_to_period_to_shift[node][period] = _cosine_distance(period_to_pos[period], centroid)

    return node_to_period_to_pos, node_to_period_to_shift

def is_converging_pair(period, n1, n2, node_to_period_to_pos, all_time=True):
    if n1 not in node_to_period_to_pos or n2 not in node_to_period_to_pos:
        return False
    c1 = node_to_period_to_pos[n1]['ALL'] if all_time else node_to_period_to_pos[n1]['<'+period]
    c2 = node_to_period_to_pos[n2]['ALL'] if all_time else node_to_period_to_pos[n2]['<'+period]
    centroid_dist = _cosine_distance(c1, c2)
    p1 = node_to_period_to_pos[n1][period][1]
    p2 = node_to_period_to_pos[n2][period][1]
    period_dist = _cosine_distance(p1, p2)
    return period_dist < centroid_dist
