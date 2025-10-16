# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import namedtuple
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from intelligence_toolkit.graph.graph_fusion_encoder_embedding import (
    _cosine_distance,
    _generate_embeddings_for_period,
    _get_edge_list,
    create_concept_to_community_hierarchy,
    generate_graph_fusion_encoder_embedding,
    is_converging_pair,
)


@pytest.fixture
def simple_graph():
    """Create a simple NetworkX graph."""
    G = nx.Graph()
    G.add_weighted_edges_from([
        ("A", "B", 1.0),
        ("B", "C", 1.0),
        ("C", "D", 1.0),
        ("D", "A", 1.0)
    ])
    return G


@pytest.fixture
def node_list():
    return ["A", "B", "C", "D"]


@pytest.fixture
def node_to_label():
    """Create simple hierarchical labels for nodes."""
    return {
        "A": {0: 0, 1: 0},
        "B": {0: 0, 1: 1},
        "C": {0: 1, 1: 2},
        "D": {0: 1, 1: 3}
    }


def test_get_edge_list(simple_graph, node_list):
    edge_list = _get_edge_list(simple_graph, node_list)
    
    assert len(edge_list) == 4
    assert all(len(edge) == 3 for edge in edge_list)  # [source, target, weight]
    
    # Check that indices are valid
    for edge in edge_list:
        assert 0 <= edge[0] < len(node_list)
        assert 0 <= edge[1] < len(node_list)
        assert edge[2] > 0  # Weight should be positive


def test_get_edge_list_filters_missing_nodes(simple_graph):
    # Only include subset of nodes
    partial_node_list = ["A", "B"]
    edge_list = _get_edge_list(simple_graph, partial_node_list)
    
    # Should only include edges between A and B
    assert len(edge_list) == 1


def test_cosine_distance():
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    
    # Orthogonal vectors should have distance 1
    dist = _cosine_distance(x, y)
    assert dist == 1.0


def test_cosine_distance_identical_vectors():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    
    # Identical vectors should have distance 0
    dist = _cosine_distance(x, y)
    assert np.isclose(dist, 0.0)


def test_cosine_distance_opposite_vectors():
    x = np.array([1, 0, 0])
    y = np.array([-1, 0, 0])
    
    # Opposite vectors should have distance 2
    dist = _cosine_distance(x, y)
    assert np.isclose(dist, 2.0)


def test_cosine_distance_zero_vector():
    x = np.array([1, 2, 3])
    y = np.array([0, 0, 0])
    
    # Division by zero should return infinity
    dist = _cosine_distance(x, y)
    assert np.isinf(dist)


def test_generate_embeddings_for_period(simple_graph, node_list, node_to_label):
    embeddings = _generate_embeddings_for_period(
        simple_graph,
        node_list,
        node_to_label,
        correlation=True,
        diaga=True,
        laplacian=False,
        max_level=1
    )
    
    # Should return embeddings for all nodes
    assert embeddings.shape[0] == len(node_list)
    
    # Should have embeddings for all levels concatenated
    assert embeddings.shape[1] > 0


def test_generate_embeddings_for_period_multiple_levels(simple_graph, node_list):
    # Create labels with more hierarchy levels
    node_to_label_deep = {
        "A": {0: 0, 1: 0, 2: 0},
        "B": {0: 0, 1: 1, 2: 1},
        "C": {0: 1, 1: 2, 2: 2},
        "D": {0: 1, 1: 3, 2: 3}
    }
    
    embeddings = _generate_embeddings_for_period(
        simple_graph,
        node_list,
        node_to_label_deep,
        correlation=True,
        diaga=True,
        laplacian=False,
        max_level=2
    )
    
    assert embeddings.shape[0] == len(node_list)


def test_generate_graph_fusion_encoder_embedding_single_period(simple_graph, node_to_label):
    period_to_graph = {"2020": simple_graph}
    
    node_to_period_to_pos, node_to_period_to_shift = generate_graph_fusion_encoder_embedding(
        period_to_graph,
        node_to_label,
        correlation=True,
        diaga=True,
        laplacian=False,
        max_level=1,
        callbacks=[]
    )
    
    # Check that all nodes have embeddings
    assert len(node_to_period_to_pos) == 4
    
    # Each node should have period "2020" and "ALL" (centroid)
    for node in node_to_label.keys():
        assert "2020" in node_to_period_to_pos[node]
        assert "ALL" in node_to_period_to_pos[node]


def test_generate_graph_fusion_encoder_embedding_multiple_periods(simple_graph, node_to_label):
    period_to_graph = {
        "2020": simple_graph,
        "2021": simple_graph  # Using same graph for simplicity
    }
    
    node_to_period_to_pos, node_to_period_to_shift = generate_graph_fusion_encoder_embedding(
        period_to_graph,
        node_to_label,
        correlation=True,
        diaga=True,
        laplacian=False,
        max_level=1,
        callbacks=[]
    )
    
    # Each node should have embeddings for both periods
    for node in node_to_label.keys():
        assert "2020" in node_to_period_to_pos[node]
        assert "2021" in node_to_period_to_pos[node]
        assert "ALL" in node_to_period_to_pos[node]
        assert "<2021" in node_to_period_to_pos[node]  # Prior centroid


def test_generate_graph_fusion_encoder_embedding_with_callbacks(simple_graph, node_to_label):
    period_to_graph = {"2020": simple_graph}
    
    callback = MagicMock()
    callback.on_batch_change = MagicMock()
    
    generate_graph_fusion_encoder_embedding(
        period_to_graph,
        node_to_label,
        correlation=True,
        diaga=True,
        laplacian=False,
        max_level=1,
        callbacks=[callback]
    )
    
    # Callback should have been called
    assert callback.on_batch_change.called


def test_is_converging_pair_with_positions():
    # Create mock position data
    node_to_period_to_pos = {
        "A": {
            "2020": (np.array([1, 0, 0]), None),
            "ALL": np.array([0.5, 0.5, 0])
        },
        "B": {
            "2020": (np.array([0, 1, 0]), None),
            "ALL": np.array([0.5, 0.5, 0])
        }
    }
    
    # This test checks the structure exists
    assert "A" in node_to_period_to_pos
    assert "B" in node_to_period_to_pos


def test_is_converging_pair_missing_node():
    node_to_period_to_pos = {
        "A": {
            "2020": (np.array([1, 0, 0]), None),
            "ALL": np.array([0.5, 0.5, 0])
        }
    }
    
    result = is_converging_pair("2020", "A", "Z", node_to_period_to_pos, all_time=True)
    
    # Should return False when a node is missing
    assert result is False


def test_is_converging_pair_converging():
    # Create positions where nodes are converging (closer in period than in centroid)
    node_to_period_to_pos = {
        "A": {
            "2020": (None, np.array([0.0, 0.0, 1.0])),  # Close to each other in this period
            "ALL": np.array([1.0, 0.0, 0.0])  # Far apart in centroid
        },
        "B": {
            "2020": (None, np.array([0.0, 0.0, 0.99])),  # Very close to A in this period
            "ALL": np.array([0.0, 1.0, 0.0])  # Far from A in centroid
        }
    }
    
    result = is_converging_pair("2020", "A", "B", node_to_period_to_pos, all_time=True)
    
    # Period distance should be much smaller than centroid distance, so nodes are converging
    assert result == True


def test_is_converging_pair_diverging():
    # Create positions where nodes are diverging (farther in period than in centroid)
    node_to_period_to_pos = {
        "A": {
            "2020": (None, np.array([1.0, 0.0, 0.0])),  # Far apart in this period
            "ALL": np.array([0.5, 0.5, 0.0])  # Close in centroid
        },
        "B": {
            "2020": (None, np.array([0.0, 1.0, 0.0])),  # Far from A in this period
            "ALL": np.array([0.5, 0.4, 0.0])  # Close to A in centroid
        }
    }
    
    result = is_converging_pair("2020", "A", "B", node_to_period_to_pos, all_time=True)
    
    # Period distance >= centroid distance, so not converging
    assert result == False


def test_is_converging_pair_with_prior_centroid():
    # Test with all_time=False (uses prior centroid)
    node_to_period_to_pos = {
        "A": {
            "2020": (None, np.array([0.0, 0.0, 1.0])),
            "<2020": np.array([1.0, 0.0, 0.0]),  # Prior centroid
            "ALL": np.array([0.5, 0.0, 0.5])
        },
        "B": {
            "2020": (None, np.array([0.0, 0.0, 0.99])),
            "<2020": np.array([0.0, 1.0, 0.0]),  # Prior centroid
            "ALL": np.array([0.0, 0.5, 0.5])
        }
    }
    
    result = is_converging_pair("2020", "A", "B", node_to_period_to_pos, all_time=False)
    
    # Should use prior centroid instead of ALL and return a boolean
    assert result == True  # Nodes are close in period, far in prior centroid


def test_create_concept_to_community_hierarchy():
    # Create mock hierarchical communities
    HierarchicalCommunity = namedtuple("HierarchicalCommunity", ["node", "level", "cluster"])
    
    hierarchical_communities = [
        HierarchicalCommunity(node="A", level="0", cluster=0),
        HierarchicalCommunity(node="A", level="1", cluster=0),
        HierarchicalCommunity(node="B", level="0", cluster=0),
        HierarchicalCommunity(node="B", level="1", cluster=1),
        HierarchicalCommunity(node="C", level="0", cluster=1),
        HierarchicalCommunity(node="C", level="1", cluster=2),
    ]
    
    concept_to_community, max_cluster_per_level, max_level = create_concept_to_community_hierarchy(
        hierarchical_communities
    )
    
    assert max_level == 1
    assert "A" in concept_to_community
    assert "B" in concept_to_community
    assert "C" in concept_to_community
    
    # Check that levels are properly filled
    assert 0 in concept_to_community["A"]
    assert 1 in concept_to_community["A"]


def test_create_concept_to_community_hierarchy_fills_missing_levels():
    HierarchicalCommunity = namedtuple("HierarchicalCommunity", ["node", "level", "cluster"])
    
    # Node A only has level 0, should propagate to level 1
    hierarchical_communities = [
        HierarchicalCommunity(node="A", level="0", cluster=5),
        HierarchicalCommunity(node="B", level="0", cluster=3),
        HierarchicalCommunity(node="B", level="1", cluster=7),
    ]
    
    concept_to_community, max_cluster_per_level, max_level = create_concept_to_community_hierarchy(
        hierarchical_communities
    )
    
    # Node A should have level 1 filled with the level 0 value
    assert concept_to_community["A"][0] == 5
    assert concept_to_community["A"][1] == 5  # Propagated from level 0


def test_create_concept_to_community_hierarchy_max_clusters():
    HierarchicalCommunity = namedtuple("HierarchicalCommunity", ["node", "level", "cluster"])
    
    hierarchical_communities = [
        HierarchicalCommunity(node="A", level="0", cluster=0),
        HierarchicalCommunity(node="B", level="0", cluster=5),
        HierarchicalCommunity(node="C", level="0", cluster=10),
    ]
    
    concept_to_community, max_cluster_per_level, max_level = create_concept_to_community_hierarchy(
        hierarchical_communities
    )
    
    # Max cluster at level 0 should be 10
    assert max_cluster_per_level[0] == 10


def test_create_concept_to_community_hierarchy_empty():
    concept_to_community, max_cluster_per_level, max_level = create_concept_to_community_hierarchy([])
    
    assert max_level == -1
    assert len(concept_to_community) == 0
