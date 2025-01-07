# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import io
from collections import defaultdict
from datetime import datetime
from enum import Enum
from json import dumps, loads

import networkx as nx
import intelligence_toolkit.query_text_data.graph_builder as graph_builder
from intelligence_toolkit.AI.text_splitter import TextSplitter
from intelligence_toolkit.query_text_data.classes import ProcessedChunks

PeriodOption = Enum("Period", "NONE DAY WEEK MONTH QUARTER YEAR")


def concert_titled_texts_to_chunks(titled_texts):
    text_to_chunks = defaultdict(list)
    splitter = TextSplitter()
    for title, text in enumerate(titled_texts.items()):
        text_chunks = splitter.split(text)
        for index, text in enumerate(text_chunks):
            chunk = {"title": title, "text_chunk": text, "chunk_id": index + 1}
            text_chunks[index] = dumps(chunk, indent=2, ensure_ascii=False)
            text_to_chunks[title] = text_chunks
    return text_to_chunks

def process_json_text(text_json, period: PeriodOption):
    def convert_to_year_quarter(datetm):
        month = datetm.month
        quarter = (month - 1) // 3 + 1
        return f"{datetm.year}-Q{quarter}"

    chunks = []
    splitter = TextSplitter()
    text_chunks = splitter.split(text_json["text"])
    for cx, chunk in enumerate(text_chunks):
        chunk_json = {"title": text_json["title"]}
        if "timestamp" in text_json and period != PeriodOption.NONE:
            timestamp = text_json["timestamp"]
            chunk_json["timestamp"] = timestamp
            period_str = ""
            # Round timestamp to the enclosing period
            datetm = datetime.fromisoformat(timestamp)
            if period == PeriodOption.DAY:
                period_str = datetm.strftime("%Y-%m-%d")
            elif period == PeriodOption.WEEK:
                period_str = datetm.strftime("%Y-%W")
            elif period == PeriodOption.MONTH:
                period_str = datetm.strftime("%Y-%m")
            elif period == PeriodOption.QUARTER:
                period_str = convert_to_year_quarter(datetm)
            elif period == PeriodOption.YEAR:
                period_str = str(datetm.year)
            chunk_json["period"] = period_str
        if "metadata" in text_json:
            chunk_json["metadata"] = text_json["metadata"]
        chunk_json["chunk_id"] = cx + 1
        chunk_json["text_chunk"] = chunk
        chunks.append(dumps(chunk_json, indent=2, ensure_ascii=False))
    return chunks


def process_json_texts(file_to_text_jsons, period: PeriodOption):
    file_to_chunks = {}
    for file, text_json in file_to_text_jsons.items():
        file_to_chunks[file] = process_json_text(text_json, period)
    return file_to_chunks


def process_chunks(
    file_to_chunks, max_cluster_size, min_edge_weight, min_node_degree, callbacks=[]
):
    period_concept_graphs = defaultdict(nx.Graph)
    period_concept_graphs["ALL"] = nx.Graph()
    node_period_counts = defaultdict(lambda: defaultdict(int))
    edge_period_counts = defaultdict(lambda: defaultdict(int))
    previous_chunk = {}
    next_chunk = {}
    concept_to_cids = defaultdict(list)
    cid_to_concepts = defaultdict(list)
    period_to_cids = defaultdict(list)
    file_cids = []
    cid_to_text = {}
    text_to_cid = {}
    chunk_id = 0
    file_to_cids = defaultdict(list)
    for file, chunks in file_to_chunks.items():
        for chunk in chunks:
            cid_to_text[chunk_id] = chunk
            text_to_cid[chunk] = chunk_id
            file_to_cids[file].append(chunk_id)
            chunk_id += 1
    for file, cids in file_to_cids.items():
        for cx, cid in enumerate(cids):
            file_cids.append((file, cid))
            if cx > 0:
                previous_chunk[cid] = cid - 1
            if cx < len(chunks) - 1:
                next_chunk[cid] = cid + 1
    for cx, (file, cid) in enumerate(file_cids):
        for cb in callbacks:
            cb.on_batch_change(cx + 1, len(file_cids))
        period = None
        chunk = cid_to_text[cid]
        try:
            chunk_json = loads(chunk)
            if "period" in chunk_json:
                period = chunk_json["period"]
        except Exception as e:
            print(e)
            pass
        periods = ["ALL"]
        period_to_cids["ALL"].append(cid)
        if period is not None:
            periods.append(period)
            period_to_cids[period].append(cid)
        graph_builder.update_concept_graph_edges(
            node_period_counts,
            edge_period_counts,
            periods,
            chunk,
            cid,
            concept_to_cids,
            cid_to_concepts,
        )

    for node, period_counts in node_period_counts.items():
        for period, count in period_counts.items():
            period_concept_graphs[period].add_node(node, count=count)
    for edge, period_counts in edge_period_counts.items():
        for period, count in period_counts.items():
            period_concept_graphs[period].add_edge(edge[0], edge[1], weight=count)

    hierarchical_communities = {}
    community_to_label = {}
    if len(period_concept_graphs["ALL"].nodes()) > 0:
        (hierarchical_communities, community_to_label) = (
            graph_builder.prepare_concept_graphs(
                period_concept_graphs,
                max_cluster_size=max_cluster_size,
                min_edge_weight=min_edge_weight,
                min_node_degree=min_node_degree,
            )
        )
    return ProcessedChunks(
        cid_to_text=cid_to_text,
        text_to_cid=text_to_cid,
        period_concept_graphs=period_concept_graphs,
        hierarchical_communities=hierarchical_communities,
        community_to_label=community_to_label,
        concept_to_cids=concept_to_cids,
        cid_to_concepts=cid_to_concepts,
        previous_cid=previous_chunk,
        next_cid=next_chunk,
        period_to_cids=period_to_cids,
        node_period_counts=node_period_counts,
        edge_period_counts=edge_period_counts,
    )
