# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Graph encoder embedding of each period graph, fused into per-node tracks.

Ports the single-level path of the legacy ``graph_fusion_encoder_embedding``
and ``GraphEncoderEmbed`` modules as pure functions over numpy and
scipy.sparse. Every node gets one position per period, an all-time centroid
(``ALL_PERIODS``) and, for every period but the first, the centroid of the
periods before it (``PRIOR_PREFIX + period``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy import sparse

from itk_patterns.models import ALL_PERIODS, PRIOR_PREFIX

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx

    from itk_patterns.models import PatternConfig, Positions, Shifts


class PeriodEmbedder(Protocol):
    """Embeds one period graph into a matrix with one row per node."""

    def embed(
        self,
        graph: nx.Graph,
        nodes: Sequence[str],
        labels: Sequence[int],
        config: PatternConfig,
    ) -> np.ndarray: ...


# -- graph encoder embedding --------------------------------------------------


def adjacency_matrix(graph: nx.Graph, nodes: Sequence[str]) -> sparse.csr_matrix:
    """Symmetric weighted adjacency of ``graph`` over ``nodes`` in that order.

    Edges touching nodes outside ``nodes`` are ignored. Missing weights count
    as ``1``.
    """
    index = {node: i for i, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for source, target, weight in graph.edges(data="weight", default=1.0):
        if source not in index or target not in index:
            continue
        i, j = index[source], index[target]
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((float(weight), float(weight)))
    n = len(nodes)
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n, n)
    )


def add_diagonal(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """``matrix + I``: a self loop of weight one on every node."""
    return sparse.csr_matrix(matrix + sparse.identity(matrix.shape[0]))


def laplacian(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Symmetric normalisation ``D^-1/2 A D^-1/2``; zero-degree nodes stay zero."""
    degree = np.asarray(matrix.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        inverse_root = np.where(degree > 0, degree**-0.5, 0.0)
    scale = sparse.diags(inverse_root, 0)
    return sparse.csr_matrix(scale @ matrix @ scale)


def encoder_embed(matrix: sparse.csr_matrix, labels: Sequence[int]) -> np.ndarray:
    """``A @ W`` where ``W[i, k] = 1 / n_k`` when node ``i`` has class ``k``.

    Classes are the non-negative label values; ``-1`` marks an unlabeled
    node, which gets a zero row in ``W``. The result has one column per class.
    """
    codes = np.asarray(labels, dtype=np.int64)
    if codes.shape[0] != matrix.shape[0]:
        msg = f"{codes.shape[0]} labels for {matrix.shape[0]} nodes"
        raise ValueError(msg)
    k = int(codes.max()) + 1 if codes.size and codes.max() >= 0 else 0
    if k == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    class_sizes = np.bincount(codes[codes >= 0], minlength=k).astype(np.float64)
    labeled = np.flatnonzero(codes >= 0)
    encoder = sparse.csr_matrix(
        (
            1.0 / class_sizes[codes[labeled]],
            (labeled, codes[labeled]),
        ),
        shape=(matrix.shape[0], k),
    )
    return np.asarray((matrix @ encoder).todense(), dtype=np.float64)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit L2 norm; zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(norms > 0, matrix / norms, 0.0)
    return np.asarray(scaled, dtype=np.float64)


def graph_encoder_embed(
    graph: nx.Graph,
    nodes: Sequence[str],
    labels: Sequence[int],
    laplacian_normalize: bool = True,
    diagonal: bool = True,
    correlation: bool = True,
) -> np.ndarray:
    """The legacy ``GraphEncoderEmbed().run`` pipeline, one row per node."""
    matrix = adjacency_matrix(graph, nodes)
    if diagonal:
        matrix = add_diagonal(matrix)
    if laplacian_normalize:
        matrix = laplacian(matrix)
    embedding = encoder_embed(matrix, labels)
    if correlation:
        embedding = row_normalize(embedding)
    return embedding


class EncoderEmbedder:
    """Default :class:`PeriodEmbedder`: the graph encoder embedding, row-normalised."""

    def embed(
        self,
        graph: nx.Graph,
        nodes: Sequence[str],
        labels: Sequence[int],
        config: PatternConfig,
    ) -> np.ndarray:
        return row_normalize(
            graph_encoder_embed(
                graph,
                nodes,
                labels,
                laplacian_normalize=config.laplacian,
                diagonal=config.diagonal,
                correlation=config.correlation,
            )
        )


# -- fusion across periods ----------------------------------------------------


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """``1 - cos(x, y)``; infinity when either vector has zero norm."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0.0:
        return float("inf")
    return float(1.0 - np.dot(x, y) / denominator)


def fuse_embeddings(
    period_graphs: dict[str, nx.Graph],
    node_labels: dict[str, int],
    embedder: PeriodEmbedder,
    config: PatternConfig,
) -> tuple[Positions, Shifts]:
    """Embed every period and derive centroids and shifts per node.

    ``positions[node][period]`` is the node's row in that period's embedding;
    ``positions[node][ALL_PERIODS]`` the mean over all periods and
    ``positions[node][PRIOR_PREFIX + period]`` the mean over the periods
    sorted before ``period``. ``shifts[node][period]`` is the cosine distance
    from the period position to the all-time centroid, and
    ``shifts[node][PRIOR_PREFIX + period]`` to the prior centroid.
    """
    nodes = sorted(node_labels)
    labels = [node_labels[node] for node in nodes]
    periods = sorted(period_graphs)
    positions: Positions = {node: {} for node in nodes}
    for period in periods:
        embedding = embedder.embed(period_graphs[period], nodes, labels, config)
        if embedding.shape[0] != len(nodes):
            msg = f"Embedder returned {embedding.shape[0]} rows for {len(nodes)} nodes"
            raise ValueError(msg)
        for index, node in enumerate(nodes):
            positions[node][period] = np.asarray(embedding[index], dtype=np.float64)
    shifts: Shifts = {node: {} for node in nodes}
    for node in nodes:
        track = positions[node]
        centroid = np.mean([track[p] for p in periods], axis=0)
        track[ALL_PERIODS] = centroid
        prior: list[np.ndarray] = []
        for period in periods:
            if prior:
                prior_centroid = np.mean(prior, axis=0)
                track[PRIOR_PREFIX + period] = prior_centroid
                shifts[node][PRIOR_PREFIX + period] = cosine_distance(
                    track[period], prior_centroid
                )
            prior.append(track[period])
            shifts[node][period] = cosine_distance(track[period], centroid)
    return positions, shifts


def is_converging_pair(
    period: str,
    node1: str,
    node2: str,
    positions: Positions,
    convergence: str = "legacy",
    all_time: bool = True,
) -> bool:
    """Whether two nodes sit closer in ``period`` than their centroids do.

    ``convergence="legacy"`` reproduces the shipped workflow, which compared
    only the second component of each period position (so two scalars, whose
    cosine distance is 0 when they share a sign, 2 when they differ and
    infinite when either is 0). ``"vector"`` compares the whole positions.
    With ``all_time`` the centroids are the all-time ones; otherwise the
    centroids of the periods before ``period``.
    """
    if node1 not in positions or node2 not in positions:
        return False
    key = ALL_PERIODS if all_time else PRIOR_PREFIX + period
    track1, track2 = positions[node1], positions[node2]
    if key not in track1 or key not in track2:
        return False
    centroid_distance = cosine_distance(track1[key], track2[key])
    p1, p2 = track1[period], track2[period]
    if convergence == "legacy":
        if p1.shape[0] < 2 or p2.shape[0] < 2:
            return False
        p1, p2 = p1[1:2], p2[1:2]
    return cosine_distance(p1, p2) < centroid_distance


__all__ = [
    "EncoderEmbedder",
    "PeriodEmbedder",
    "add_diagonal",
    "adjacency_matrix",
    "cosine_distance",
    "encoder_embed",
    "fuse_embeddings",
    "graph_encoder_embed",
    "is_converging_pair",
    "laplacian",
    "row_normalize",
]
