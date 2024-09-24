# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import re
from collections import defaultdict

import nltk
import numpy as np
import networkx as nx
from graspologic import partition
from nltk.data import find
from textblob import TextBlob


def download_if_not_exists(resource_name) -> None:
    try:
        find(resource_name)
    except LookupError:
        nltk.download(resource_name)


download_if_not_exists("brown")
download_if_not_exists("punkt")
download_if_not_exists("punkt_tab")


def update_concept_graph_edges(node_to_period_counts, edge_to_period_counts, periods, chunk, cid, concept_to_cids, cid_to_concepts):
    nps = sorted(set(TextBlob(chunk).noun_phrases))
    filtered_nps = []
    for np in nps:
        parts = np.split()
        if all([re.match(r"[a-zA-Z0-9\-]+", part) for part in parts]):
            filtered_nps.append(np)
    filtered_nps = sorted(filtered_nps)
    for np in filtered_nps:
        concept_to_cids[np].append(cid)
    cid_to_concepts[cid] = filtered_nps
    for period in periods:
        for np in filtered_nps:
            node_to_period_counts[np][period] += 1
        for ix, np1 in enumerate(filtered_nps):
            for np2 in filtered_nps[ix + 1 :]:
                edge_to_period_counts[(np1, np2)][period] += 1


def prepare_concept_graphs(period_concept_graphs, max_cluster_size, min_edge_weight, min_node_degree):
    prepare_concept_graph(period_concept_graphs["ALL"], min_edge_weight, min_node_degree)
    community_to_concepts, concept_to_community, hierarchical_communities = (
        detect_concept_communities(period_concept_graphs["ALL"], max_cluster_size)
    )
    for period, G in period_concept_graphs.items():
        for node in list(G.nodes()):
            if node not in concept_to_community:
                G.remove_node(node)
        for component in list(nx.connected_components(G)):
            highest_degree_node = max(component, key=lambda x: G.degree(x))
            G.add_edge(highest_degree_node, 'dummynode', weight=1)
        for node, data in G.nodes(data=True):
            data['community'] = concept_to_community[node] if node in concept_to_community else -1

    
    return community_to_concepts, concept_to_community, hierarchical_communities
    
def build_meta_graph(G, hierarchical_communities):
    level_to_communities = {}
    level_to_label_to_network = defaultdict(dict)
    max_level = max([hc.level for hc in hierarchical_communities])
    level_community_nodes = {}
    for level in range(max_level+1):
        filtered_nodes = [hc for hc in hierarchical_communities if hc.level == level]
        community_nodes = defaultdict(set)
        for hc in filtered_nodes:
            community_nodes[hc.cluster].add(hc.node)
        level_community_nodes[level] = community_nodes
        sorted_communities = sorted(community_nodes.keys(), key=lambda x: len(community_nodes[x]), reverse=True)
        level_to_communities[level] = sorted_communities
    for level, community_list in level_to_communities.items():
        for c in community_list:
            c_nodes = level_community_nodes[level][c]
            S = nx.subgraph(G, c_nodes)
            top_nodes = sorted(c_nodes, key=lambda x: G.degree(x), reverse=True)[:7]
            label = "; ".join(top_nodes)
            level_to_label_to_network[level][label] = S
    return level_to_label_to_network


def prepare_concept_graph(G, min_edge_weight, min_node_degree, std_trim=4):
    degrees = [x[1] for x in G.degree()]
    mean_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    cutoff = mean_degree + std_trim * std_degree
    G.remove_nodes_from([n for n, d in G.degree() if d > cutoff])
    G.remove_edges_from(
        [
            (u, v)
            for u, v, d in G.edges(data=True)
            if u in G.nodes() and v in G.nodes() and d["weight"] < min_edge_weight
        ]
    )
    G.remove_nodes_from([n for n, d in G.degree() if d < min_node_degree])
    for component in list(nx.connected_components(G)):
        highest_degree_node = max(component, key=lambda x: G.degree(x))
        G.add_edge(highest_degree_node, 'dummynode', weight=1)


def detect_concept_communities(G, max_cluster_size):
    clustering = partition.hierarchical_leiden(G, max_cluster_size, random_seed=42)
    node_to_community = clustering.final_level_hierarchical_clustering()
    community_to_nodes = defaultdict(list)
    for node, community in node_to_community.items():
        community_to_nodes[community].append(node)
    # sort nodes
    for community in community_to_nodes.keys():
        community_to_nodes[community] = sorted(community_to_nodes[community])
    return community_to_nodes, node_to_community, clustering
