# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Candidate pair generation by cosine nearest neighbours, numpy only."""

from __future__ import annotations

import numpy as np

Candidate = tuple[int, int, float]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def candidate_pairs(
    embeddings: np.ndarray,
    max_distance: float,
    neighbors: int = 50,
    chunk_size: int = 1024,
) -> list[Candidate]:
    """Return (left, right, cosine_distance) with left < right, sorted by distance.

    For each row the ``neighbors`` closest other rows are considered and kept
    when their cosine distance is at most ``max_distance``. Works for any row
    count, including fewer rows than ``neighbors``.
    """
    n = embeddings.shape[0]
    if n < 2:
        return []
    k = min(neighbors, n - 1)
    unit = _normalise(embeddings.astype(np.float32, copy=False))

    found: dict[tuple[int, int], float] = {}
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sims = unit[start:stop] @ unit.T
        rows = np.arange(start, stop)
        sims[np.arange(stop - start), rows] = -np.inf  # drop self-matches
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for offset, row in enumerate(rows):
            for col in top[offset]:
                distance = float(1.0 - sims[offset, col])
                if distance <= max_distance:
                    key = (min(int(row), int(col)), max(int(row), int(col)))
                    found[key] = min(distance, found.get(key, distance))

    return sorted(
        ((left, right, distance) for (left, right), distance in found.items()),
        key=lambda c: (c[2], c[0], c[1]),
    )
