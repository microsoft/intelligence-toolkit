# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import pandas as pd
import pytest

from python.attribute_patterns.detection_functions import (
    _calculate_cosine_distance,
    _calculate_euclidean_distance,
    _compute_node_pair_distances,
    _create_centroid_dists,
    create_period_shifts,
)


def test_calculate_cosine_distance():
    # Test case 1: vec1 and vec2 are orthogonal
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    assert _calculate_cosine_distance(vec1, vec2) == 1.0

    # Test case 2: vec1 and vec2 are parallel
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([2, 0, 0])
    assert _calculate_cosine_distance(vec1, vec2) == 0.0

    # Test case 3: vec1 and vec2 are not orthogonal or parallel
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    assert _calculate_cosine_distance(vec1, vec2) == pytest.approx(0.025, abs=1e-3)


def test_calculate_euclidean_distance():
    vec1 = np.array([0.1, 0.2, 0.3])
    vec2 = np.array([0.4, 0.5, 0.6])
    expected_distance = np.linalg.norm(vec1 - vec2)
    calculated_distance = _calculate_euclidean_distance(vec1, vec2)
    assert calculated_distance == expected_distance


def test_compute_node_pair_distances():
    period = "period1"
    attribute_period_embeddings = {
        "period1": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "period2": [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
    }
    sorted_nodes = ["node1", "node2", "node3"]
    node_to_ix = {"node1": 0, "node2": 1, "node3": 2}

    expected_distances = {
        ("node1", "node2"): (0.025368153802923787, 0.5196152422706632),
        ("node1", "node3"): (0.0405880544333298, 1.0392304845413265),
        ("node2", "node3"): (0.001809107314273306, 0.5196152422706632),
    }

    distances = _compute_node_pair_distances(
        period, attribute_period_embeddings, sorted_nodes, node_to_ix
    )
    assert distances == expected_distances


def test_create_centroid_dists():
    node_to_centroid = {"node1": [1, 2, 3], "node2": [4, 5, 6], "node3": [7, 8, 9]}

    expected_distances = {
        ("node1", "node2"): (0.025368153802923787, 5.196152422706632),
        ("node1", "node3"): (0.04058805443332969, 10.392304845413264),
        ("node2", "node3"): (0.001809107314273084, 5.196152422706632),
    }

    distances = _create_centroid_dists(node_to_centroid)
    assert distances == expected_distances


# def test_create_period_shifts(mocker):
#     node_to_centroid = {"node1": [1, 2, 3], "node2": [4, 5, 6]}
#     attribute_period_embeddings = {
#         "period1": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
#         "period2": [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
#     }
#     attribute_dynamic_df = pd.DataFrame({"Period": ["period1", "period2"]})
#     create_centroid_dists_mock = mocker.patch(
#         "python.attribute_patterns.detection_functions._create_centroid_dists"
#     )
#     create_centroid_dists_mock.return_value = {("node1", "node2"): (0.3, 0.4)}

#     result = create_period_shifts(
#         node_to_centroid, attribute_period_embeddings, attribute_dynamic_df
#     )

#     expected_result = {
#         "period1": {("node1", "node2"): (0.2746318461970762, -0.11961524227066322)},
#         "period2": {("node1", "node2"): (0.29149992451511525, -0.11961524227066311)},
#     }

#     assert result == expected_result
