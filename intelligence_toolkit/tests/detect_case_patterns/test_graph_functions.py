# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import pandas as pd
import pytest

from intelligence_toolkit.detect_case_patterns.config import (
    min_edge_weight,
    missing_edge_prop,
)
from intelligence_toolkit.detect_case_patterns.graph_functions import (
    convert_edge_df_to_graph,
    create_edge_df_from_atts,
)


def test_convert_edge_df_to_graph_default():
    # Create a sample edge DataFrame
    edge_df = pd.DataFrame(
        {
            "source": ["A", "B", "C", "A"],
            "target": ["B", "C", "D", "A"],
            "weight": [1, 2, 3, 3],
        }
    )

    # Call the function
    G, lcc = convert_edge_df_to_graph(edge_df)

    # Check if the graph is of type nx.Graph
    assert isinstance(G, nx.Graph)

    # Check if the largest connected component is correct
    expected_lcc = {"A", "B", "C", "D"}
    assert set(lcc) == expected_lcc


def test_convert_edge_df_to_graph_more_nodes():
    # Create a sample edge DataFrame
    edge_df = pd.DataFrame(
        {
            "source": [1, 2, 3, 4],
            "target": [2, 3, 4, 5],
            "weight": [0.5, 0.6, 0.7, 0.8],
        }
    )

    # Call the function
    G, lcc = convert_edge_df_to_graph(edge_df)

    # Check if the graph is of type nx.Graph
    assert isinstance(G, nx.Graph)

    # Check if the largest connected component is correct
    expected_lcc = {1, 2, 3, 4, 5}
    assert set(lcc) == expected_lcc


@pytest.fixture()
def sample_input_data():
    # Generate sample input data
    all_atts = ["A", "B", "C", "D"]
    pdf = pd.DataFrame(
        {"Full Attribute": [["A", "B"], ["B", "C"], ["C", "B"], ["B", "A"]]}
    )
    return all_atts, pdf


def test_create_edge_df_from_atts(sample_input_data):
    # Call the function with the sample input data
    all_atts, pdf = sample_input_data
    edge_df = create_edge_df_from_atts(
        all_atts, pdf, min_edge_weight, missing_edge_prop
    )

    # Assert that the output DataFrame has the correct columns
    assert set(edge_df.columns) == {"edge", "count", "source", "target", "weight"}

    # Assert that the output DataFrame is not empty
    assert not edge_df.empty

    assert not (edge_df["weight"] > 1).any()
    # assert not (edge_df["weight"] < 0.001).any()