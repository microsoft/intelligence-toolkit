from collections import defaultdict
from typing import Any

import networkx as nx
import polars as pl
from sklearn.neighbors import NearestNeighbors

import toolkit.risk_networks.config as config
from toolkit.AI.base_embedder import BaseEmbedder
from toolkit.AI.openai_configuration import OpenAIConfiguration
from toolkit.AI.openai_embedder import OpenAIEmbedder
from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.risk_networks.config import (
    ENTITY_LABEL,
    SIMILARITY_THRESHOLD_MAX,
    SIMILARITY_THRESHOLD_MIN,
)


async def index_nodes(
    indexed_node_types: list[str],
    main_graph: nx.Graph,
    callbacks: list[ProgressBatchCallback] | None = None,
    functions_embedder: BaseEmbedder | None = None,
    openai_configuration: OpenAIConfiguration | None = None,
    save_cache=True,
):
    if len(indexed_node_types) == 0:
        msg = "No node types to index"
        raise ValueError(msg)
    text_types = [
        (n, d["type"])
        for n, d in main_graph.nodes(data=True)
        if d["type"] in indexed_node_types
    ]
    texts = [t[0] for t in text_types]

    if functions_embedder is None:
        functions_embedder = OpenAIEmbedder(openai_configuration, config.cache_name)
    embeddings = await functions_embedder.embed_store_many(
        texts,
        callbacks,
        save_cache,
    )

    vals = [(n, t, e) for (n, t), e in zip(text_types, embeddings, strict=False)]
    edf = pl.DataFrame(vals, schema=["text", "type", "vector"])

    edf = edf.filter(pl.col("text").is_in(texts))
    embedded_texts = edf["text"].to_list()
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

    for ix in range(len(embedded_texts)):
        if progress_callbacks:
            for cb in progress_callbacks:
                cb.on_batch_change(ix, len(embedded_texts), "Infering links...")

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


def create_inferred_links(inferred_links: defaultdict[Any, set]) -> list[tuple]:
    return [
        (text, n) for text, near in inferred_links.items() for n in near if text < n
    ]


def build_inferred_df(inferred_links_list: defaultdict[set]) -> pl.DataFrame:
    link_list = [
        (text, n)
        for text, near in inferred_links_list.items()
        for n in near
        if text < n
    ]
    print("link_listlink_list", link_list)
    inferred_df = pl.DataFrame(link_list, schema=["text", "similar"])
    inferred_df = inferred_df.with_columns(
        [
            pl.col("text").str.replace(ENTITY_LABEL + ATTRIBUTE_VALUE_SEPARATOR, ""),
            pl.col("similar").str.replace(ENTITY_LABEL + ATTRIBUTE_VALUE_SEPARATOR, ""),
        ]
    )

    return inferred_df.sort(["text", "similar"])


async def index_and_infer(
    indexed_node_types: list[str],
    main_graph: nx.Graph,
    network_similarity_threshold: float,
    callbacks: list[ProgressBatchCallback] | None = None,
    functions_embedder: BaseEmbedder | None = None,
    openai_configuration: OpenAIConfiguration | None = None,
    save_cache=True,
) -> tuple[defaultdict[set], int]:
    if not len(main_graph.nodes()):
        msg = "Graph is empty"
        raise ValueError(msg)

    (
        embedded_texts,
        nearest_text_distances,
        nearest_text_indices,
    ) = await index_nodes(
        indexed_node_types,
        main_graph,
        callbacks,
        functions_embedder,
        openai_configuration,
        save_cache,
    )

    inferred_links = infer_nodes(
        network_similarity_threshold,
        embedded_texts,
        nearest_text_indices,
        nearest_text_distances,
        callbacks,
    )

    return inferred_links, len(embedded_texts)
