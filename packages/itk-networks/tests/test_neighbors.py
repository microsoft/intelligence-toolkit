# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import pytest
from itk_core.progress import ProgressBatchCallback
from itk_networks.neighbors import (
    BruteForceCosine,
    SklearnNeighbors,
    infer_links,
    link_pairs,
    links_frame,
    nearest_neighbors,
    symmetric,
)

VECTORS = np.array([
    [1.0, 0.0],
    [1.0, 0.1],  # close to row 0
    [0.0, 1.0],
    [0.0, 2.0],  # same direction as row 2
])


def test_brute_force_returns_self_first_and_sorted_distances():
    distances, indices = BruteForceCosine().nearest(VECTORS, 3)
    assert distances.shape == indices.shape == (4, 3)
    assert indices[:, 0].tolist() == [0, 1, 2, 3]
    assert np.all(np.diff(distances, axis=1) >= -1e-12)
    assert indices[0, 1] == 1
    assert distances[2, 1] == pytest.approx(0.0, abs=1e-12)
    assert indices[2, 1] == 3


def test_brute_force_matches_manual_cosine_distance():
    distances, indices = BruteForceCosine().nearest(VECTORS, 2)
    expected = 1 - (1.0 / np.sqrt(1.01))
    assert distances[0, 1] == pytest.approx(expected)
    assert indices[0, 1] == 1


def test_brute_force_handles_zero_vectors():
    distances, _ = BruteForceCosine().nearest(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
    assert np.isfinite(distances).all()


def test_nearest_neighbors_caps_k_at_row_count():
    distances, indices = nearest_neighbors(VECTORS, neighbors=20)
    assert distances.shape == (4, 4)
    assert indices.shape == (4, 4)


def test_nearest_neighbors_empty():
    distances, indices = nearest_neighbors(np.zeros((0, 0)))
    assert distances.shape == (0, 0)
    assert indices.shape == (0, 0)


def test_infer_links_is_symmetric_and_thresholded():
    texts = ["a", "b", "c", "d"]
    distances, indices = nearest_neighbors(VECTORS)
    tight = infer_links(texts, distances, indices, 0.001)
    assert tight == {"c": frozenset({"d"}), "d": frozenset({"c"})}
    loose = infer_links(texts, distances, indices, 0.01)
    assert loose["a"] == frozenset({"b"})
    assert loose["b"] == frozenset({"a"})
    assert "a" not in loose["c"]
    everything = infer_links(texts, distances, indices, 1.0)
    assert everything["a"] == frozenset({"b", "c", "d"})


def test_infer_links_ignores_identical_texts():
    distances, indices = nearest_neighbors(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert infer_links(["same", "same"], distances, indices, 0.5) == {}


@pytest.mark.parametrize("threshold", [0.0, 0.0001, 1.5])
def test_infer_links_validates_threshold(threshold):
    with pytest.raises(ValueError, match=r"between 0\.001 and 1\.0"):
        infer_links(["a"], np.zeros((1, 1)), np.zeros((1, 1), int), threshold)


def test_infer_links_reports_progress():
    callback = ProgressBatchCallback()
    seen = []
    callback.on_batch_change = lambda current, total, message="": seen.append((
        current,
        total,
    ))
    distances, indices = nearest_neighbors(VECTORS)
    infer_links(["a", "b", "c", "d"], distances, indices, 0.5, [callback])
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_symmetric_adds_reverse_direction_and_drops_self():
    links = symmetric({"a": frozenset({"b", "a"}), "c": frozenset()})
    assert links == {"a": frozenset({"b"}), "b": frozenset({"a"})}


def test_link_pairs_and_frame_strip_entity_prefix():
    links = symmetric({"ENTITY==Acme": frozenset({"ENTITY==Acme Co", "ENTITY==Abc"})})
    assert link_pairs(links) == [
        ("ENTITY==Abc", "ENTITY==Acme"),
        ("ENTITY==Acme", "ENTITY==Acme Co"),
    ]
    frame = links_frame(links)
    assert frame.columns == ["text", "similar"]
    assert frame.rows() == [("Abc", "Acme"), ("Acme", "Acme Co")]


def test_sklearn_and_brute_force_infer_the_same_links_on_float64():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    base = rng.normal(size=(12, 8))
    # Near-duplicate rows with exactly tied distances, like count vectors have.
    vectors = np.vstack([base, base[:4] * 2.0, base[:2] * 3.0])
    texts = [f"t{i}" for i in range(vectors.shape[0])]
    expected = link_pairs(
        infer_links(texts, *BruteForceCosine().nearest(vectors, 5), 0.05)
    )
    actual = link_pairs(
        infer_links(texts, *SklearnNeighbors().nearest(vectors, 5), 0.05)
    )
    assert vectors.dtype == np.float64
    assert actual == expected
    assert len(expected) == 8  # rows 0 and 1 form triples, rows 2 and 3 pairs
