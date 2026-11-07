# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
from itk_match.blocking import candidate_pairs


def _unit(*rows):
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_candidate_pairs_returns_close_pairs_once_sorted_by_distance():
    embeddings = _unit([1, 0, 0], [1, 0.01, 0], [0, 1, 0], [0, 1, 0.02])
    pairs = candidate_pairs(embeddings, max_distance=0.01)

    assert [(left, right) for left, right, _ in pairs] == [(0, 1), (2, 3)]
    assert all(left < right for left, right, _ in pairs)
    assert pairs[0][2] <= pairs[1][2]


def test_candidate_pairs_respects_max_distance():
    embeddings = _unit([1, 0], [0, 1])
    assert candidate_pairs(embeddings, max_distance=0.5) == []
    assert len(candidate_pairs(embeddings, max_distance=1.0)) == 1


def test_candidate_pairs_works_with_fewer_rows_than_neighbors():
    embeddings = _unit([1, 0], [1, 0.001], [1, 0.002])
    pairs = candidate_pairs(embeddings, max_distance=0.1, neighbors=50)
    assert len(pairs) == 3


def test_candidate_pairs_limits_neighbors():
    rng = np.random.default_rng(0)
    base = np.ones((6, 4), dtype=np.float32)
    embeddings = base + rng.normal(scale=1e-3, size=base.shape).astype(np.float32)
    pairs = candidate_pairs(embeddings, max_distance=1.0, neighbors=1)
    # Each row contributes at most one pair; mutual nearest pairs collapse.
    assert 3 <= len(pairs) <= 6


def test_candidate_pairs_chunks_consistently():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(40, 8)).astype(np.float32)
    whole = candidate_pairs(embeddings, max_distance=0.9, neighbors=5)
    chunked = candidate_pairs(embeddings, max_distance=0.9, neighbors=5, chunk_size=7)
    assert [(left, right) for left, right, _ in whole] == [
        (left, right) for left, right, _ in chunked
    ]
    # BLAS kernels can round float32 dot products differently across chunk sizes.
    np.testing.assert_allclose(
        [distance for _, _, distance in whole],
        [distance for _, _, distance in chunked],
        rtol=0.0,
        atol=1e-6,
    )


def test_candidate_pairs_trivial_inputs():
    assert candidate_pairs(np.zeros((0, 3)), 1.0) == []
    assert candidate_pairs(np.ones((1, 3)), 1.0) == []
