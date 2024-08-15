# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict
import pdfplumber
import io
import re
from textblob import TextBlob
import networkx as nx
from graspologic import partition
from app.util.wkhtmltopdf import config_pdfkit, pdfkit_options # should move this into the lib
import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')

def process_files(input_file_bytes, text_splitter, callbacks=[]):
    text_to_chunks = defaultdict(list)
    for fx, file_name in enumerate(input_file_bytes.keys()):
        for cb in callbacks:
            cb.on_batch_change(fx+1, len(input_file_bytes.keys()))
        bytes = input_file_bytes[file_name]

        if file_name.endswith(".pdf"):
            page_texts = []
            pdf_reader = pdfplumber.open(io.BytesIO(bytes))
            for px in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[px].extract_text()
                page_texts.append(page_text)
            doc_text = ' '.join(page_texts)
        else:
            doc_text = bytes.decode("utf-8")
        doc_chunks = text_splitter.split(doc_text)
        text_to_chunks[file_name] = doc_chunks

    return text_to_chunks

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

def process_chunks(text_to_chunks, embedder, embedding_cache, max_cluster_size, callbacks=[]):
    G = nx.Graph()
    previous_chunk = {}
    next_chunk = {}
    concept_to_chunks = defaultdict(list)
    chunk_to_concepts = defaultdict(list)
    text_to_vectors = defaultdict(list)
    file_chunks = []
    for file, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
            file_chunks.append((file, chunk))
            if cx > 0:
                previous_chunk[chunk] = chunks[cx-1]
            if cx < len(chunks) - 1:
                next_chunk[chunk] = chunks[cx+1]
    for cx, (file, chunk) in enumerate(file_chunks):
        for cb in callbacks:
            cb.on_batch_change(cx+1, len(file_chunks))
        formatted_chunk = chunk #.replace("\n", " ")
        chunk_vec = embedder.embed_store_one(
            formatted_chunk, embedding_cache
        )
        text_to_vectors[file].append(np.array(chunk_vec))
        update_concept_graph(G, chunk, concept_to_chunks, chunk_to_concepts)
    clean_concept_graph(G, 2, 1)
    community_to_concepts, concept_to_community = detect_concept_communities(G, max_cluster_size)
    return text_to_chunks, text_to_vectors, G, community_to_concepts, concept_to_community, concept_to_chunks, chunk_to_concepts, previous_chunk, next_chunk