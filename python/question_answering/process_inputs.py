# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict
import pdfplumber
import io
import networkx as nx
from app.util.wkhtmltopdf import config_pdfkit, pdfkit_options
import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')
import python.question_answering.graph_functions as graph_functions
from python.AI.text_splitter import TextSplitter

def process_file_bytes(input_file_bytes, callbacks=[]):
    text_to_chunks = defaultdict(list)
    splitter = TextSplitter()
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
        doc_chunks = splitter.split(doc_text)
        text_to_chunks[file_name] = doc_chunks
    return text_to_chunks

def process_texts(text_dict):
    text_to_chunks = defaultdict(list)
    splitter = TextSplitter()
    for title, text in text_dict.items():
        text_chunks = splitter.split(text)
        text_to_chunks[title] = text_chunks
    return text_to_chunks

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
        graph_functions.update_concept_graph(G, chunk, concept_to_chunks, chunk_to_concepts)
    graph_functions.clean_concept_graph(G, 2, 1)
    community_to_concepts, concept_to_community = graph_functions.detect_concept_communities(G, max_cluster_size)
    return text_to_vectors, G, community_to_concepts, concept_to_community, concept_to_chunks, chunk_to_concepts, previous_chunk, next_chunk