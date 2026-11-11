# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import math

import networkx as nx
import numpy as np
import pytest
from itk_patterns import PatternConfig
from itk_patterns.embed import (
    EncoderEmbedder,
    add_diagonal,
    adjacency_matrix,
    cosine_distance,
    encoder_embed,
    fuse_embeddings,
    graph_encoder_embed,
    is_converging_pair,
    laplacian,
    row_normalize,
)
from itk_patterns.models import ALL_PERIODS, PRIOR_PREFIX
from itk_patterns_testing import FakeEmbedder


@pytest.fixture()
def square() -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        ("A", "B", 1.0),
        ("B", "C", 2.0),
        ("C", "D", 1.0),
        ("D", "A", 1.0),
    ])
    return graph


NODES = ["A", "B", "C", "D"]


def test_adjacency_matrix_is_symmetric_and_weighted(square):
    matrix = adjacency_matrix(square, NODES).toarray()
    assert matrix.shape == (4, 4)
    assert np.array_equal(matrix, matrix.T)
    assert matrix[1, 2] == 2.0
    assert matrix[0, 2] == 0.0


def test_adjacency_matrix_ignores_nodes_outside_the_list(square):
    matrix = adjacency_matrix(square, ["A", "B"]).toarray()
    assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_adjacency_matrix_does_not_depend_on_edge_direction():
    forward = nx.Graph()
    forward.add_weighted_edges_from([("A", "B", 1.0), ("B", "C", 3.0)])
    backward = nx.Graph()
    backward.add_weighted_edges_from([("C", "B", 3.0), ("B", "A", 1.0)])
    assert np.array_equal(
        adjacency_matrix(forward, ["A", "B", "C"]).toarray(),
        adjacency_matrix(backward, ["A", "B", "C"]).toarray(),
    )


def test_add_diagonal_puts_ones_on_the_diagonal(square):
    matrix = add_diagonal(adjacency_matrix(square, NODES)).toarray()
    assert np.array_equal(np.diag(matrix), np.ones(4))


def test_laplacian_normalises_symmetrically_and_keeps_zero_degree_rows(square):
    matrix = adjacency_matrix(square, [*NODES, "Z"])
    normalised = laplacian(matrix).toarray()
    degree = matrix.toarray().sum(axis=0)
    expected = degree[0] ** -0.5 * 1.0 * degree[1] ** -0.5
    assert normalised[0, 1] == pytest.approx(expected)
    assert np.all(normalised[4] == 0.0)
    assert np.all(normalised[:, 4] == 0.0)


def test_encoder_embed_averages_neighbours_per_class(square):
    matrix = adjacency_matrix(square, NODES)
    embedding = encoder_embed(matrix, [0, 1, 0, 1])
    assert embedding.shape == (4, 2)
    # A links to B and D (class 1, two members, weights 1 and 1): 1/2 + 1/2
    assert embedding[0].tolist() == [0.0, 1.0]
    # B links to A (class 0, weight 1) and C (class 0, weight 2): (1 + 2) / 2
    assert embedding[1].tolist() == [1.5, 0.0]


def test_encoder_embed_treats_minus_one_as_unlabeled(square):
    matrix = adjacency_matrix(square, NODES)
    embedding = encoder_embed(matrix, [0, -1, 0, -1])
    assert embedding.shape == (4, 1)
    # B's only class-0 neighbours are A and C, class 0 has two members
    assert embedding[1, 0] == pytest.approx(1.5)
    # A's neighbours are both unlabeled
    assert embedding[0, 0] == 0.0


def test_encoder_embed_rejects_label_count_mismatch(square):
    with pytest.raises(ValueError, match="labels for"):
        encoder_embed(adjacency_matrix(square, NODES), [0, 1])


def test_row_normalize_unit_rows_and_zero_rows():
    rows = row_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert rows[0].tolist() == [0.6, 0.8]
    assert rows[1].tolist() == [0.0, 0.0]


def test_graph_encoder_embed_rows_have_unit_norm_with_correlation(square):
    embedding = graph_encoder_embed(square, NODES, [0, 1, 0, 1])
    assert embedding.shape == (4, 2)
    assert np.allclose(np.linalg.norm(embedding, axis=1), 1.0)
    plain = graph_encoder_embed(square, NODES, [0, 1, 0, 1], correlation=False)
    assert not np.allclose(np.linalg.norm(plain, axis=1), 1.0)


def test_encoder_embedder_follows_config_flags(square):
    config = PatternConfig(period_column="p", laplacian=False, diagonal=False)
    embedding = EncoderEmbedder().embed(square, NODES, [0, 1, 0, 1], config)
    expected = row_normalize(
        graph_encoder_embed(
            square, NODES, [0, 1, 0, 1], laplacian_normalize=False, diagonal=False
        )
    )
    assert np.allclose(embedding, expected)


def test_cosine_distance_cases():
    assert cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(
        0
    )
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        1
    )
    assert cosine_distance(np.array([1.0]), np.array([-1.0])) == pytest.approx(2)
    assert math.isinf(cosine_distance(np.array([0.0]), np.array([1.0])))


def test_fuse_embeddings_centroids_priors_and_shifts(prepared, config):
    fake = FakeEmbedder()
    positions, shifts = fuse_embeddings(
        prepared.period_graphs, prepared.node_labels, fake, config
    )
    assert fake.calls == ["P1", "P2", "P3"]
    track = positions["city=A"]
    assert set(track) == {"P1", "P2", "P3", ALL_PERIODS, "<P2", "<P3"}
    assert track[ALL_PERIODS].tolist() == pytest.approx([3.0, -4.0 / 3.0])
    assert track[PRIOR_PREFIX + "P3"].tolist() == [3.0, -4.0]
    assert shifts["city=A"]["P1"] == pytest.approx(
        cosine_distance(track["P1"], track[ALL_PERIODS])
    )
    assert shifts["city=A"]["<P3"] == pytest.approx(
        cosine_distance(track["P3"], track["<P3"])
    )
    assert "<P1" not in shifts["city=A"]
    # a node that never changes sits on its centroid
    assert shifts["city=B"]["P2"] == pytest.approx(0.0)


def test_fuse_embeddings_rejects_wrong_row_count(prepared, config):
    class Short:
        def embed(self, graph, nodes, labels, config):
            return np.zeros((1, 2))

    with pytest.raises(ValueError, match="rows for"):
        fuse_embeddings(prepared.period_graphs, prepared.node_labels, Short(), config)


def test_is_converging_pair_legacy_uses_second_component_sign(embedded):
    positions = embedded.positions
    assert positions is not None
    # both positive in P3, centroids differ
    assert is_converging_pair("P3", "city=A", "product=X", positions)
    # same sign in P1 too, so the period distance is 0 while centroids differ
    assert is_converging_pair("P1", "city=A", "product=X", positions)
    # identical tracks share a centroid: distance 0 is not below 0
    assert not is_converging_pair("P3", "city=B", "product=Y", positions)
    # opposite signs in P3: distance 2 exceeds any centroid distance
    assert not is_converging_pair("P3", "city=A", "city=B", positions)


def test_is_converging_pair_prior_centroids():
    positions = {
        "a": {"P1": np.array([1.0, 1.0]), "P2": np.array([1.0, 1.0])},
        "b": {"P1": np.array([1.0, 1.0]), "P2": np.array([1.0, 1.0])},
    }
    positions["a"][PRIOR_PREFIX + "P2"] = np.array([1.0, 0.0])
    positions["b"][PRIOR_PREFIX + "P2"] = np.array([0.0, 1.0])
    # no prior centroid exists for the first period
    assert not is_converging_pair("P1", "a", "b", positions, all_time=False)
    # distance 0 in P2 against prior centroids 1 apart
    assert is_converging_pair("P2", "a", "b", positions, all_time=False)


def test_is_converging_pair_vector_mode_compares_whole_positions():
    positions = {
        "a": {
            "P1": np.array([1.0, 0.0]),
            "P2": np.array([1.0, 1.0]),
            ALL_PERIODS: np.array([1.0, 0.5]),
        },
        "b": {
            "P1": np.array([0.0, 1.0]),
            "P2": np.array([1.0, 1.0]),
            ALL_PERIODS: np.array([0.5, 1.0]),
        },
    }
    # P1: orthogonal (distance 1) while the centroids are 0.2 apart
    assert not is_converging_pair("P1", "a", "b", positions, convergence="vector")
    # P2: identical (distance 0)
    assert is_converging_pair("P2", "a", "b", positions, convergence="vector")
    # legacy mode only sees the second components: both positive in P1? no, a is 0
    assert not is_converging_pair("P1", "a", "b", positions)
    assert is_converging_pair("P2", "a", "b", positions)


def test_is_converging_pair_unknown_nodes_and_short_vectors():
    positions = {
        "a": {"P1": np.array([1.0]), ALL_PERIODS: np.array([1.0])},
        "b": {"P1": np.array([1.0]), ALL_PERIODS: np.array([0.5])},
    }
    assert not is_converging_pair("P1", "a", "zzz", positions)
    assert not is_converging_pair("P1", "a", "b", positions)
