# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from collections import defaultdict
from typing import Any

import networkx as nx
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from python.AI.embedder import Embedder
from python.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from python.helpers.progress_batch_callback import ProgressBatchCallback
from python.risk_networks.constants import (
    SIMILARITY_THRESHOLD_MAX,
    SIMILARITY_THRESHOLD_MIN,
)

from . import config


def _merge_condition(x, y) -> bool:
    """
    Merge condition function for merging nodes in the graph.
    """
    x_parts = set(x.split(config.list_sep))
    y_parts = set(y.split(config.list_sep))
    return any(
        x_part.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
        == y_part.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
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
) -> nx.Graph:
    G = nx.Graph()
    value_to_atts = defaultdict(set)
    for link_list in network_attribute_links:
        for link in link_list:
            n1 = f"{config.entity_label}{ATTRIBUTE_VALUE_SEPARATOR}{link[0]}"
            n2 = f"{link[1]}{ATTRIBUTE_VALUE_SEPARATOR}{link[2]}"
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            G.add_edge(edge[0], edge[1], type=link[1])
            G.add_node(n1, type=config.entity_label)
            G.add_node(n2, type=link[1])
            value_to_atts[link[2]].add(n2)

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
    trimmed_attributes,
    entity_to_community_ix,
    inferred_links,
    integrated_flags,
) -> tuple[nx.Graph, Any]:
    N = nx.Graph()
    trimmed_nodeset = trimmed_attributes["Attribute"].unique().tolist()
    for node in nodes:
        n_c = (
            str(entity_to_community_ix[node]) if node in entity_to_community_ix else ""
        )
        N.add_node(node, type=config.entity_label, network=n_c, flags=0)
        ent_neighbors = set(G.neighbors(node)).union(inferred_links[node])
        for ent_neighbor in ent_neighbors:
            if ent_neighbor in trimmed_nodeset:
                continue

            if ent_neighbor.startswith(config.entity_label) and node != ent_neighbor:
                en_c = entity_to_community_ix.get(ent_neighbor, "")
                N.add_node(ent_neighbor, type=config.entity_label, network=en_c)
                N.add_edge(node, ent_neighbor)
            else:  # att
                N.add_node(
                    ent_neighbor,
                    type=ent_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                    flags=0,
                )
                N.add_edge(node, ent_neighbor)
                att_neighbors = set(G.neighbors(ent_neighbor)).union(
                    inferred_links[ent_neighbor]
                )
                for att_neighbor in att_neighbors:
                    if att_neighbor in trimmed_nodeset or not att_neighbor.startswith(
                        config.entity_label
                    ):
                        continue

                    N.add_node(
                        att_neighbor,
                        type=att_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                        flags=0,
                    )
                    fuzzy_att_neighbors = set(G.neighbors(att_neighbor)).union(
                        inferred_links[att_neighbor]
                    )
                    for fuzzy_att_neighbor in fuzzy_att_neighbors:
                        if (
                            fuzzy_att_neighbor in trimmed_nodeset
                            or fuzzy_att_neighbor.startswith(config.entity_label)
                        ):
                            continue
                        N.add_node(
                            fuzzy_att_neighbor,
                            type=fuzzy_att_neighbor.split(ATTRIBUTE_VALUE_SEPARATOR)[0],
                            flags=0,
                        )
                        N.add_edge(att_neighbor, fuzzy_att_neighbor)
    if len(integrated_flags) > 0:
        fdf = integrated_flags
        fdf = fdf[fdf["count"] > 0]
        flagged_nodes = fdf["qualified_entity"].unique().tolist()
        for node in flagged_nodes:
            if node in N.nodes():
                N.nodes[node]["flags"] = fdf.loc[
                    fdf["qualified_entity"] == node, "count"
                ].sum()
    return N, integrated_flags  # change sv???


def index_nodes(
    indexed_node_types: list[str],
    overall_graph: nx.Graph,
    callbacks: list[ProgressBatchCallback] | None = None,
    use_local=False,
    save_cache=True,
):
    if len(indexed_node_types) == 0:
        msg = "No node types to index"
        raise ValueError(msg)
    text_types = [
        (n, d["type"])
        for n, d in overall_graph.nodes(data=True)
        if d["type"] in indexed_node_types
    ]
    texts = [t[0] for t in text_types]

    functions_embedder = Embedder(None, config.cache_dir, use_local)
    embeddings = functions_embedder.embed_store_many(
        texts,
        callbacks,
        save_cache,
    )

    vals = [(n, t, e) for (n, t), e in zip(text_types, embeddings, strict=False)]
    edf = pd.DataFrame(vals, columns=["text", "type", "vector"])

    edf = edf[edf["text"].isin(texts)]
    embedded_texts = edf["text"].tolist()
    nbrs = NearestNeighbors(
        n_neighbors=20,
        n_jobs=1,
        algorithm="auto",
        leaf_size=20,
        metric="cosine",
    ).fit(embeddings)

    (
        nearest_text_distances,
        nearest_text_indices,
    ) = nbrs.kneighbors(embeddings)

    return embedded_texts, nearest_text_distances, nearest_text_indices


def create_links(inferred_links: defaultdict[Any, set]) -> list[tuple]:
    return [
        (text, n) for text, near in inferred_links.items() for n in near if text < n
    ]


def infer_nodes(
    similarity_threshold: float,
    embedded_texts: list[str],
    nearest_text_indices: list[list[int]],
    nearest_text_distances: list[list[float]],
    progress_callbacks: list[ProgressBatchCallback] | None = None,
) -> defaultdict[Any, set]:
    inferred_links = defaultdict(set)
    if (
        similarity_threshold < SIMILARITY_THRESHOLD_MIN
        or similarity_threshold > SIMILARITY_THRESHOLD_MAX
    ):
        msg = f"Similarity threshold must be between {SIMILARITY_THRESHOLD_MIN} and {SIMILARITY_THRESHOLD_MAX}"
        raise ValueError(msg)
    # pb = st.progress(0, text="Inferring links...")
    for ix in range(len(embedded_texts)):
        if progress_callbacks:
            for cb in progress_callbacks:
                cb.on_batch_change(ix, len(embedded_texts))
        # pb.progress(int(ix * 100 / len(texts)), text="Inferring links...")
        near_is = nearest_text_indices[ix]
        near_ds = nearest_text_distances[ix]
        nearest = zip(near_is, near_ds, strict=False)
        for near_i, near_d in nearest:
            if (near_i != ix and near_d <= similarity_threshold) and embedded_texts[
                ix
            ] != embedded_texts[near_i]:
                inferred_links[embedded_texts[ix]].add(embedded_texts[near_i])
                inferred_links[embedded_texts[near_i]].add(embedded_texts[ix])

    return inferred_links