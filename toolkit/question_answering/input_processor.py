# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import io
from collections import defaultdict
from json import dumps

import networkx as nx
import numpy as np
import pdfplumber

import toolkit.question_answering.graph_builder as graph_builder
from toolkit.AI.text_splitter import TextSplitter


def process_file_bytes(input_file_bytes, callbacks=[]):
    text_to_chunks = defaultdict(list)
    splitter = TextSplitter()
    for fx, file_name in enumerate(input_file_bytes.keys()):
        for cb in callbacks:
            cb.on_batch_change(fx + 1, len(input_file_bytes.keys()))
        bytes = input_file_bytes[file_name]

        if file_name.endswith(".pdf"):
            page_texts = []
            pdf_reader = pdfplumber.open(io.BytesIO(bytes))
            for px in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[px].extract_text()
                page_texts.append(page_text)
            doc_text = " ".join(page_texts)
        else:
            doc_text = bytes.decode("utf-8")
        text_chunks = splitter.split(doc_text)
        for cx, chunk in enumerate(text_chunks):
            text_to_chunks[file_name].append(
                dumps(
                    {"title": file_name, "chunk_id": cx + 1, "text_chunk": chunk},
                    indent=2,
                )
            )
    return text_to_chunks


def process_json_texts(text_jsons):
    """
    Texts are represented as JSON objects with title, text, and (optional) timestamp and metadata fields
    """
    text_to_chunks = defaultdict(list)
    splitter = TextSplitter()
    for text_json in text_jsons:
        text_chunks = splitter.split(text_json["text"])
        for cx, chunk in enumerate(text_chunks):
            chunk_json = {"title": text_json["title"]}
            if "timestamp" in text_json:
                chunk_json["timestamp"] = text_json["timestamp"]
            if "metadata" in text_json:
                chunk_json["metadata"] = text_json["metadata"]
            chunk_json["chunk_id"] = cx + 1
            chunk_json["text_chunk"] = chunk
            text_to_chunks[text_json["title"]].append(dumps(chunk_json, indent=2))
    return text_to_chunks


def process_chunks(
    text_to_chunks, embedder, embedding_cache, max_cluster_size, callbacks=[]
):
    concept_graph = nx.Graph()
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
                previous_chunk[chunk] = chunks[cx - 1]
            if cx < len(chunks) - 1:
                next_chunk[chunk] = chunks[cx + 1]
    for cx, (file, chunk) in enumerate(file_chunks):
        for cb in callbacks:
            cb.on_batch_change(cx + 1, len(file_chunks))
        formatted_chunk = chunk  # .replace("\n", " ")
        chunk_vec = embedder.embed_store_one(formatted_chunk, embedding_cache)
        text_to_vectors[file].append(np.array(chunk_vec))
        graph_builder.update_concept_graph(
            concept_graph, chunk, concept_to_chunks, chunk_to_concepts
        )
    graph_builder.clean_concept_graph(concept_graph, 2, 1)
    community_to_concepts, concept_to_community = (
        graph_builder.detect_concept_communities(concept_graph, max_cluster_size)
    )
    return (
        text_to_vectors,
        concept_graph,
        community_to_concepts,
        concept_to_community,
        concept_to_chunks,
        chunk_to_concepts,
        previous_chunk,
        next_chunk,
    )
