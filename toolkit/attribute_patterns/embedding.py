# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import pandas as pd

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR

from .config import correlation, diaga, laplacian, type_val_sep
from .graph_encoder_embed import GraphEncoderEmbed


def get_node_mappings(df):
    """Create mappings for nodes to indices and labels."""
    node_list = sorted(df["Full Attribute"].unique().tolist())
    sorted_att_types = sorted(df["Attribute Type"].unique())
    node_to_ix = {n: i for i, n in enumerate(node_list)}
    node_to_label = {
        n: sorted_att_types.index(n.split(type_val_sep)[0]) for n in node_list
    }
    return node_to_ix, node_to_label


def get_edge_list(graph, node_list, node_to_ix):
    """Generate a list of edges with weights for existing nodes."""
    return [
        [node_to_ix[s], node_to_ix[t], w]
        for s, t, w in graph.edges(data="weight")
        if s in node_list and t in node_list
    ]


def generate_embeddings_for_period(graph, node_list, node_to_ix, node_to_label):
    """Generate embeddings for a single period."""
    edge_list = get_edge_list(graph, node_list, node_to_ix)
    num_nodes = len(node_list)
    labels = np.array([node_to_label[node] for node in node_list]).reshape(
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


def serialize_embeddings(period_embeddings, period):
    """Serialize embeddings to DataFrame."""
    embedding_rows = []
    for node_id in range(len(period_embeddings[period])):
        row = [
            f"{node_id}=={period}",
            period,
            period_embeddings[period][node_id].tolist(),
        ]
        embedding_rows.append(row)
    columns = ["dynamic_node_id", "period", "embedding"]
    return pd.DataFrame(embedding_rows, columns=columns)


def calculate_centroids(period_embeddings, node_list, node_to_ix, time_to_graph_keys):
    """Calculate the centroid of embeddings for each node."""
    node_to_centroid = {}
    for node in node_list:
        node_to_centroid[node] = np.mean(
            [
                period_embeddings[period][node_to_ix[node]]
                for period in time_to_graph_keys
            ],
            axis=0,
        ).tolist()
    return node_to_centroid


def generate_embedding(df, time_to_graph):
    """Generate embeddings for all periods and calculate centroids."""
    node_to_ix, node_to_label = get_node_mappings(df)
    node_list = list(node_to_ix.keys())

    period_embeddings = {}
    embedding_dfs = []

    for period, graph in time_to_graph.items():
        period_embeddings[period] = generate_embeddings_for_period(
            graph, node_list, node_to_ix, node_to_label
        )
        embedding_df = serialize_embeddings(period_embeddings, period)
        embedding_dfs.append(embedding_df)

    overall_embedding_df = pd.concat(embedding_dfs)
    node_to_centroid = calculate_centroids(
        period_embeddings, node_list, node_to_ix, time_to_graph.keys()
    )
    overall_embedding_df["Full Attribute"] = overall_embedding_df[
        "dynamic_node_id"
    ].apply(lambda x: node_list[int(x.split(ATTRIBUTE_VALUE_SEPARATOR)[0])])

    return overall_embedding_df, node_to_centroid, period_embeddings
