# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from intelligence_toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from intelligence_toolkit.graph.graph_encoder_embed import GraphEncoderEmbed


def _get_edge_list(graph, node_list):
    """Generate a list of edges with weights for existing nodes."""
    node_to_ix = {node: i for i, node in enumerate(node_list)}
    return [
        [node_to_ix[s], node_to_ix[t], w]
        for s, t, w in graph.edges(data="weight")
        if s in node_list and t in node_list
    ]


def _generate_embeddings_for_period(
    graph, node_list, node_to_label, correlation, diaga, laplacian, max_level
):
    """Generate embeddings for a single period."""
    edge_list = _get_edge_list(graph, node_list)
    num_nodes = len(node_list)

    ############TODO#########################
    node_to_ix = {node: i for i, node in enumerate(node_list)}

    # Note that this function relies upon the incoming node_to_label dictionary to be FULL and complete.  Every node MUST be defined for EVERY level in the hierarchy.
    # When a node doesn't technically exist, make sure that it is populated with the leaf level parent for the logic below to work.

    level_embeddings = {}
    # For each level
    for level in range(0, max_level + 1):
        labels = np.array(
            [
                node_to_label[node][level] if node in node_to_label else -1
                for node in node_list
            ]
        ).reshape(
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
        level_embeddings[level] = Z

    # Now create a joint embedding across all levels
    # get the length of any vector at the root level - this is the minimal number of dimensions to PCA to
    # and resize all vectors to be of this same length.
    # TODO: Experiment with different balances of each layer (intuition is we need to weight higher levels in the hierarchy with more weight from observations)
    embedding_length = level_embeddings[0].shape[1]

    normalized_vectors = {}

    for level in range(0, max_level + 1):
        # First check to see if we should PCA the whole thing first to a standard dimensionality
        # Obviously for root level 0 - nothing needs to be done
        if level_embeddings[level].shape[1] == embedding_length:
            # at the root level, just copy the vectors over
            # TODO: normalize

            normalized_vectors[level] = normalize(level_embeddings[level].toarray())

        else:
            # ideally we actually run a PCA, but that doesn't scale - so we instead use a TSVD
            tsvd = TruncatedSVD(n_components=embedding_length)

            tsvd.fit(level_embeddings[level].toarray())
            # TODO: normalize vectors

            normalized_vectors[level] = normalize(
                tsvd.transform(level_embeddings[level].toarray())
            )

    concat_vectors = {}
    # Next, ONLY copy over the nodes that actually exist at a given level
    for node in node_list:
        for level in range(0, max_level + 1):
            if level not in concat_vectors:
                concat_vectors[level] = {}
            # Check to see if the node actually existed natively at this level of the hierarchy - otherwise we can take alternative logic - like zeroing out this part of the vector.
            if level == 0:
                # First, all nodes exist at 0
                concat_vectors[level][node] = normalized_vectors[level][
                    node_to_ix[node]
                ]
            else:
                # Deeper the level 0, we have to check if the node actually exists at this level
                if node_to_label[node][level - 1] != node_to_label[node][level]:
                    # the node existed at this depth of the hierarchy
                    # TODO: Check to make sure we're indexing
                    concat_vectors[level][node] = normalized_vectors[level][
                        node_to_ix[node]
                    ]
                else:
                    # if the node has the SAME cluster ID as its parent, then we know it doesn't actually at this level in the hierarchy
                    # So this zeros out the vector if it didn't exist at level of the hierarchy.... we can zero it out OR we can use the cluster membership from the tiers above and then keep the embedding
                    concat_vectors[level][node] = [0] * embedding_length

    nodevecs = []
    # next - concat all the vectors together for all layers of the hierarchy
    for node in node_list:
        nodevec = []
        for level in range(0, max_level + 1):
            nodevec = np.append(nodevec, concat_vectors[level][node])
        nodevecs.append(np.array(nodevec))
    nodearry = np.vstack(nodevecs)

    return nodearry


def _cosine_distance(x, y):
    den = np.linalg.norm(x) * np.linalg.norm(y)
    dist = 1 - (np.dot(x, y) / den) if den > 0 else np.inf
    return dist


def generate_graph_fusion_encoder_embedding(
    period_to_graph,
    node_to_label,
    correlation,
    diaga,
    laplacian,
    max_level,
    callbacks=[],
):
    """Generate embeddings for all periods and calculate centroids.
    All-time centroids are encoded as 'ALL' and prior centroids are encoded as '<'+period.
    """
    node_list = sorted(node_to_label.keys())
    node_to_period_to_pos = defaultdict(lambda: defaultdict(np.array))
    node_to_period_to_shift = defaultdict(lambda: defaultdict(np.array))
    for period, graph in period_to_graph.items():
        period_embedding = _generate_embeddings_for_period(
            graph, node_list, node_to_label, correlation, diaga, laplacian, max_level
        )
        for node_id in range(len(period_embedding)):
            node_to_period_to_pos[node_list[node_id]][period] = period_embedding[
                node_id
            ]

    for ix, (node, period_to_pos) in enumerate(node_to_period_to_pos.items()):
        for callback in callbacks:
            callback.on_batch_change(ix + 1, len(node_to_period_to_pos.keys()))
        all_positions = [pos for period, pos in period_to_pos.items()]
        centroid = np.mean(all_positions, axis=0)
        node_to_period_to_pos[node]["ALL"] = centroid
        sorted_periods = sorted(period_to_pos.keys())
        prior_positions = []
        for period in sorted_periods:
            if len(prior_positions) > 0:
                # Encodes prior centroid position and shift
                prior_centroid = np.mean(prior_positions, axis=0)
                node_to_period_to_pos[node]["<" + period] = prior_centroid
                node_to_period_to_shift[node]["<" + period] = _cosine_distance(
                    period_to_pos[period], prior_centroid
                )
            prior_positions.append(period_to_pos[period])
        node_to_period_to_shift[node][period] = _cosine_distance(
            period_to_pos[period], centroid
        )

    return node_to_period_to_pos, node_to_period_to_shift


def is_converging_pair(period, n1, n2, node_to_period_to_pos, all_time=True):
    if n1 not in node_to_period_to_pos or n2 not in node_to_period_to_pos:
        return False
    c1 = (
        node_to_period_to_pos[n1]["ALL"]
        if all_time
        else node_to_period_to_pos[n1]["<" + period]
    )
    c2 = (
        node_to_period_to_pos[n2]["ALL"]
        if all_time
        else node_to_period_to_pos[n2]["<" + period]
    )
    centroid_dist = _cosine_distance(c1, c2)
    p1 = node_to_period_to_pos[n1][period][1]
    p2 = node_to_period_to_pos[n2][period][1]
    period_dist = _cosine_distance(p1, p2)
    return period_dist < centroid_dist


def create_concept_to_community_hierarchy(hierarchical_communities):
    concept_to_community_hierarchy = {}
    max_level = -1

    for rw in hierarchical_communities:
        if rw.node not in concept_to_community_hierarchy:
            concept_to_community_hierarchy[rw.node] = {}
        concept_to_community_hierarchy[rw.node][int(rw.level)] = rw.cluster
        if int(rw.level) > max_level:
            max_level = int(rw.level)

    # Clean up hierarchy to propagate leaf nodes down to all lower levels in the hierarchy
    max_cluster_per_level = {}
    for level_cluster in range(0, max_level + 1):
        max_cluster_per_level[level_cluster] = -1
    for ky in concept_to_community_hierarchy:
        lastobservedcluster = -1
        #    if len(concept_to_community_hierarchy[ky]) < (max_level + 1):
        for level in range(0, max_level + 1):
            # Always start with 0 and go deeper - that way we'll always have a fallback since everything has membership for zero at the least
            if level in concept_to_community_hierarchy[ky]:
                lastobservedcluster = concept_to_community_hierarchy[ky][level]
            else:
                # These are the tiers that are missing clusters
                concept_to_community_hierarchy[ky][level] = lastobservedcluster
            if lastobservedcluster > max_cluster_per_level[level]:
                max_cluster_per_level[level] = lastobservedcluster
    return concept_to_community_hierarchy, max_cluster_per_level, max_level