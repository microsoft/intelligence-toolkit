# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Nearest neighbours over node embeddings, and the links inferred from them."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl

from itk_networks.models import (
    DEFAULT_NEIGHBORS,
    SIMILARITY_THRESHOLD_MAX,
    SIMILARITY_THRESHOLD_MIN,
)
from itk_networks.text import unqualify_entity

if TYPE_CHECKING:
    from itk_core.progress import ProgressBatchCallback

LINKS_SCHEMA = {"text": pl.Utf8, "similar": pl.Utf8}


class NeighborIndex(Protocol):
    """Finds the ``k`` nearest rows of ``vectors`` for every row, itself included."""

    def nearest(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, indices)`` of shape ``(n, k)``, nearest first."""
        ...


class BruteForceCosine:
    """Exact cosine neighbours in numpy; the default, with no sklearn dependency."""

    def nearest(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = matrix / norms
        distances = np.maximum(1.0 - unit @ unit.T, 0.0)
        np.fill_diagonal(distances, -1.0)  # self sorts first, like sklearn
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        nearest = np.take_along_axis(distances, order, axis=1)
        return np.maximum(nearest, 0.0), order


class SklearnNeighbors:
    """The legacy scikit-learn index, for parity checks. Requires scikit-learn."""

    def __init__(self, leaf_size: int = 20) -> None:
        self.leaf_size = leaf_size

    def nearest(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(
            n_neighbors=k,
            n_jobs=1,
            algorithm="auto",
            leaf_size=self.leaf_size,
            metric="cosine",
        ).fit(vectors)
        return index.kneighbors(vectors)


def nearest_neighbors(
    embeddings: np.ndarray,
    index: NeighborIndex | None = None,
    neighbors: int = DEFAULT_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Distances and indices of the nearest ``neighbors`` rows, capped at ``n``."""
    count = embeddings.shape[0] if embeddings.ndim == 2 else 0
    if count == 0:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=np.int64)
    k = min(neighbors, count)
    return (index or BruteForceCosine()).nearest(embeddings, k)


def check_threshold(threshold: float) -> None:
    if not SIMILARITY_THRESHOLD_MIN <= threshold <= SIMILARITY_THRESHOLD_MAX:
        msg = (
            "Similarity threshold must be between "
            f"{SIMILARITY_THRESHOLD_MIN} and {SIMILARITY_THRESHOLD_MAX}"
        )
        raise ValueError(msg)


def infer_links(
    texts: list[str] | tuple[str, ...],
    distances: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    callbacks: list[ProgressBatchCallback] | None = None,
) -> dict[str, frozenset[str]]:
    """Symmetric links between texts whose cosine distance is within ``threshold``."""
    check_threshold(threshold)
    links: defaultdict[str, set[str]] = defaultdict(set)
    total = len(texts)
    for ix, text in enumerate(texts):
        for cb in callbacks or ():
            cb.on_batch_change(ix, total, "Infering links...")
        for near_i, near_d in zip(indices[ix], distances[ix], strict=True):
            other = texts[int(near_i)]
            if int(near_i) != ix and near_d <= threshold and other != text:
                links[text].add(other)
                links[other].add(text)
    return {text: frozenset(near) for text, near in links.items()}


def symmetric(links: dict[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    """The same links with every edge present in both directions."""
    out: defaultdict[str, set[str]] = defaultdict(set)
    for text, near in links.items():
        for other in near:
            if other != text:
                out[text].add(other)
                out[other].add(text)
    return {text: frozenset(near) for text, near in out.items()}


def link_pairs(links: dict[str, frozenset[str]]) -> list[tuple[str, str]]:
    """Each undirected link once, as ``(smaller, larger)``, sorted."""
    return sorted(
        (text, other) for text, near in links.items() for other in near if text < other
    )


def links_frame(links: dict[str, frozenset[str]]) -> pl.DataFrame:
    """``text, similar`` pairs with the ``ENTITY==`` prefix removed, sorted."""
    pairs = link_pairs(links)
    return pl.DataFrame(
        {
            "text": [unqualify_entity(a) for a, _ in pairs],
            "similar": [unqualify_entity(b) for _, b in pairs],
        },
        schema=LINKS_SCHEMA,
    ).sort(["text", "similar"])
