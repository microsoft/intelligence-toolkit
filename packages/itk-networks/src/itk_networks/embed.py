# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Embed graph nodes of selected types into an aligned matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from itk_core.utils import hash_text

from itk_networks.models import TYPE

if TYPE_CHECKING:
    import networkx as nx
    from itk_core.classes import VectorData
    from itk_core.progress import ProgressBatchCallback


class Embedder(Protocol):
    """The slice of itk_core.base_embedder.BaseEmbedder this package relies on."""

    async def embed_store_many(
        self,
        data: list[VectorData],
        callbacks: list[ProgressBatchCallback] | None = None,
        cache_data: bool = True,
    ) -> Any: ...


def nodes_to_embed(graph: nx.Graph, node_types: list[str]) -> list[tuple[str, str]]:
    """``(node, type)`` pairs for nodes of the given types, sorted by node."""
    if not node_types:
        msg = "No node types to index"
        raise ValueError(msg)
    wanted = set(node_types)
    return sorted(
        (node, data[TYPE])
        for node, data in graph.nodes(data=True)
        if data.get(TYPE) in wanted
    )


async def embed_nodes(
    graph: nx.Graph,
    node_types: list[str],
    embedder: Embedder,
    callbacks: list[ProgressBatchCallback] | None = None,
    cache: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Return the sorted node texts and a float32 matrix with one row each."""
    pairs = nodes_to_embed(graph, node_types)
    texts = [node for node, _ in pairs]
    if not texts:
        return [], np.zeros((0, 0), dtype=np.float32)
    payload: list[Any] = [
        {"hash": hash_text(node), "text": node, "additional_details": {"type": kind}}
        for node, kind in pairs
    ]
    stored = await embedder.embed_store_many(payload, callbacks, cache)
    by_text = {
        item["text"]: np.asarray(item["vector"], dtype=np.float32) for item in stored
    }
    missing = [t for t in texts if t not in by_text]
    if missing:
        msg = f"Embedder returned no vector for {len(missing)} node(s)"
        raise ValueError(msg)
    return texts, np.vstack([by_text[t] for t in texts])
