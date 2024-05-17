# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from python.attribute_patterns.functions import (
    calculate_cosine_distance,
    calculate_euclidean_distance,
    compute_node_pair_distances,
    convert_edge_df_to_graph,
    create_edge_df_from_atts,
)
from python.attribute_patterns.model import prepare_graph


def test_convert_edge_df_to_graph_default():
    # Create a sample edge DataFrame
    edge_df = pd.DataFrame({
        'source': ['A', 'B', 'C', 'A'],
        'target': ['B', 'C', 'D', 'A'],
        'weight': [1, 2, 3,3]
    })

    # Call the function
    G, lcc = convert_edge_df_to_graph(edge_df)

    # Check if the graph is of type nx.Graph
    assert isinstance(G, nx.Graph)

    # Check if the largest connected component is correct
    expected_lcc = {'A', 'B', 'C', 'D'}
    assert set(lcc) == expected_lcc

def test_convert_edge_df_to_graph_more_nodes():
    # Create a sample edge DataFrame
    edge_df = pd.DataFrame({
        'source': [1, 2, 3, 4],
        'target': [2, 3, 4, 5],
        'weight': [0.5, 0.6, 0.7, 0.8]
    })

    # Call the function
    G, lcc = convert_edge_df_to_graph(edge_df)

    # Check if the graph is of type nx.Graph
    assert isinstance(G, nx.Graph)

    # Check if the largest connected component is correct
    expected_lcc = {1,2,3,4,5}
    assert set(lcc) == expected_lcc

@pytest.fixture
def sample_input_data():
    # Generate sample input data
    all_atts = ['A', 'B', 'C', 'D']
    pdf = pd.DataFrame({'Full Attribute': [['A', 'B'], ['B', 'C'], ['C', 'D'], ['A', 'C'], ['B', 'D']]})
    mi = True  # You can change this based on your testing requirements
    return all_atts, pdf, mi

def test_create_edge_df_from_atts(sample_input_data):
    # Call the function with the sample input data
    all_atts, pdf, mi = sample_input_data
    edge_df = create_edge_df_from_atts(all_atts, pdf, mi)

    # Assert that the output DataFrame has the correct columns
    assert set(edge_df.columns) == {'edge','count','source', 'target', 'weight'}

    # Assert that the output DataFrame is not empty
    assert not edge_df.empty

    # Assert that the weight column has values between the specified range
    assert (edge_df['weight'] <= 1).all()
    assert (edge_df['weight'] >= 0.0001).all()

def test_create_edge_df_from_atts_mi_false(sample_input_data):
    # Call the function with the sample input data
    all_atts, pdf, mi = sample_input_data
    mi = False
    edge_df = create_edge_df_from_atts(all_atts, pdf, mi)

    # Assert that the output DataFrame has the correct columns
    assert set(edge_df.columns) == {'edge','count','source', 'target', 'weight'}

    # Assert that the output DataFrame is not empty
    assert not edge_df.empty

    assert len(edge_df['weight'].isna()) > 1

def test_prepare_graph(mocker):
    create_edge_df_from_atts_mock = mocker.patch("python.attribute_patterns.functions.create_edge_df_from_atts")
    edge_df = pd.DataFrame({
        'source': ['A', 'B', 'C', 'A'],
        'target': ['B', 'C', 'D', 'A'],
        'weight': [1, 2, 3, 3]
    })
    create_edge_df_from_atts_mock.return_value = edge_df
    df = pd.DataFrame({'Subject ID':[1,2,2,1],'Period':[2020,2021,2021,2020],'Full Attribute': ['ab=1', 'bc=2','ab=2', 'bc=1']})
    pdf, time_to_graph = prepare_graph(df)
    assert 'Grouping ID' in pdf.columns
    assert pdf['Grouping ID'].str.contains('@').all()
    assert all(isinstance(graph, nx.classes.graph.Graph) for graph in time_to_graph.values())
    assert len(time_to_graph) == 2

def test_calculate_cosine_distance():
    # Test case 1: vec1 and vec2 are orthogonal
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    assert calculate_cosine_distance(vec1, vec2) == 1.0

    # Test case 2: vec1 and vec2 are parallel
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([2, 0, 0])
    assert calculate_cosine_distance(vec1, vec2) == 0.0

    # Test case 3: vec1 and vec2 are not orthogonal or parallel
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    assert calculate_cosine_distance(vec1, vec2) == pytest.approx(0.025, abs=1e-3)


def test_calculate_euclidean_distance():
    vec1 = np.array([0.1, 0.2, 0.3])
    vec2 = np.array([0.4, 0.5, 0.6])
    expected_distance = np.linalg.norm(vec1 - vec2)
    calculated_distance = calculate_euclidean_distance(vec1, vec2)
    assert calculated_distance == expected_distance

def test_compute_node_pair_distances():
    period = 'period1'
    attribute_period_embeddings = {
        'period1': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        'period2': [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]
    }
    sorted_nodes = ['node1', 'node2', 'node3']
    node_to_ix = {'node1': 0, 'node2': 1, 'node3': 2}

    expected_distances = {('node1', 'node2'): (0.025368153802923787, 0.5196152422706632), ('node1', 'node3'): (0.0405880544333298, 1.0392304845413265), ('node2', 'node3'): (0.001809107314273306, 0.5196152422706632)}


    distances = compute_node_pair_distances(period, attribute_period_embeddings, sorted_nodes, node_to_ix)
    assert distances == expected_distances
