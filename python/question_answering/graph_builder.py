# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict
import re
from textblob import TextBlob
from graspologic import partition
import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')

def update_concept_graph(G, chunk, concept_to_chunks, chunk_to_concepts):
    nps = sorted(set(TextBlob(chunk).noun_phrases))
    filtered_nps = []
    for np in nps:
        parts = np.split()
        if all([re.match(r'[a-zA-Z0-9\-]+', part) for part in parts]):
            filtered_nps.append(np)
    
    for np in filtered_nps:
        concept_to_chunks[np].append(chunk)
    chunk_to_concepts[chunk] = filtered_nps
    for ix, np1 in enumerate(filtered_nps):
        for np2 in filtered_nps[ix+1:]:
            if G.has_edge(np1, np2):
                G[np1][np2]["weight"] += 1
            else:
                G.add_edge(np1, np2, weight=1)
    for np in filtered_nps:
        if np not in G.nodes:
            G.add_node(np, count=1)
        else:
            old_count = G.nodes[np]['count'] if 'count' in G.nodes[np] else 0
            G.nodes[np]['count'] = old_count + 1

def clean_concept_graph(G, min_edge_weight, min_node_degree):
    degrees = [x[1] for x in G.degree()]
    mean_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    cutoff = mean_degree + 4*std_degree
    G.remove_nodes_from([
        n for n, d in G.degree() if d > cutoff
    ])
    G.remove_edges_from([
        (u, v) for u, v, d in G.edges(data=True) if u in G.nodes() and v in G.nodes() and d["weight"] < min_edge_weight
    ])
    G.remove_nodes_from([
        n for n, d in G.degree() if d < min_node_degree
    ])

def detect_concept_communities(G, max_cluster_size):
    clustering = partition.hierarchical_leiden(G, max_cluster_size, random_seed=42)
    node_to_community = clustering.final_level_hierarchical_clustering()
    community_to_nodes = defaultdict(list)
    for node, community in node_to_community.items():
        community_to_nodes[community].append(node)
    return community_to_nodes, node_to_community
