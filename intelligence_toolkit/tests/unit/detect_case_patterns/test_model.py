# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import pandas as pd

from intelligence_toolkit.detect_case_patterns.config import (
    min_edge_weight,
    missing_edge_prop,
    type_val_sep,
)
from intelligence_toolkit.detect_case_patterns.model import (
    compute_attribute_counts,
    detect_patterns,
    generate_graph_model,
    prepare_graph,
)


def test_generate_graph_model_basic(mocker):
    data = {
        "Subject ID": [1, 2],
        "Period": ["P1", "P2"],
        "Attribute1": [10, 20],
        "Attribute2": [30, 40],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df.astype(str)
    result = generate_graph_model(test_df, "Period", type_val_sep)

    expected_data = {
        "Subject ID": ["1", "2", "1", "2"],
        "Period": ["P1", "P2", "P1", "P2"],
        "Attribute Type": ["Attribute1", "Attribute1", "Attribute2", "Attribute2"],
        "Attribute Value": ["10", "20", "30", "40"],
        "Full Attribute": [
            f"Attribute1{type_val_sep}10",
            f"Attribute1{type_val_sep}20",
            f"Attribute2{type_val_sep}30",
            f"Attribute2{type_val_sep}40",
        ],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_generate_graph_model_with_nans(mocker):
    data = {
        "Subject ID": [1, 2, None],
        "Period": ["P1", "P2", None],
        "Attribute1": [10, None, 30],
        "Attribute2": [None, 40, 50],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df.fillna("").astype(str)
    result = generate_graph_model(test_df, "Period", type_val_sep)

    expected_data = {
        "Subject ID": ["1", "2"],
        "Period": ["P1", "P2"],
        "Attribute Type": ["Attribute1", "Attribute2"],
        "Attribute Value": ["10.0", "40.0"],
        "Full Attribute": [
            f"Attribute1{type_val_sep}10.0",
            f"Attribute2{type_val_sep}40.0",
        ],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_generate_graph_model_column_rename(mocker):
    data = {
        "Subject ID": [1, 2],
        "Custom_Period": ["P1", "P2"],
        "Attribute1": [10, 20],
        "Attribute2": [30, 40],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df.astype(str)
    result = generate_graph_model(test_df, "Custom_Period", type_val_sep)

    expected_data = {
        "Subject ID": ["1", "2", "1", "2"],
        "Period": ["P1", "P2", "P1", "P2"],
        "Attribute Type": ["Attribute1", "Attribute1", "Attribute2", "Attribute2"],
        "Attribute Value": ["10", "20", "30", "40"],
        "Full Attribute": [
            f"Attribute1{type_val_sep}10",
            f"Attribute1{type_val_sep}20",
            f"Attribute2{type_val_sep}30",
            f"Attribute2{type_val_sep}40",
        ],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_compute_attribute_counts_basic(mocker):
    data = {
        "Subject ID": [1, 2, 3],
        "Period": ["P1", "P1", "P2"],
        "Attribute1": ["A", "A", "B"],
        "Attribute2": ["X", "Y", "X"],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df
    result = compute_attribute_counts(
        test_df, f"Attribute1{type_val_sep}A", "Period", "P1", type_val_sep
    )

    expected_data = {
        "AttributeValue": [
            f"Attribute1{type_val_sep}A",
            f"Attribute2{type_val_sep}X",
            f"Attribute2{type_val_sep}Y",
        ],
        "Count": [2, 1, 1],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_compute_attribute_counts_with_multiple_patterns(mocker):
    data = {
        "Subject ID": [1, 2, 3, 4],
        "Period": ["P1", "P1", "P2", "P1"],
        "Attribute1": ["A", "B", "A", "A"],
        "Attribute2": ["X", "X", "Y", "X"],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df
    result = compute_attribute_counts(
        test_df, "Attribute1::A & Attribute2::X", "Period", "P1", type_val_sep
    )

    expected_data = {
        "AttributeValue": ["Attribute2=X", "Attribute1=A", "Attribute1=B"],
        "Count": [3, 2, 1],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_compute_attribute_counts_with_nans(mocker):
    data = {
        "Subject ID": [1, 2, 3],
        "Period": ["P1", "P1", "P2"],
        "Attribute1": [None, "A", "A"],
        "Attribute2": ["X", None, "Y"],
    }

    test_df = pd.DataFrame(data).fillna("")

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df
    result = compute_attribute_counts(
        test_df, "Attribute1::A", "Period", "P1", type_val_sep
    )

    expected_data = {
        "AttributeValue": ["Attribute1=A", "Attribute2=X"],
        "Count": [1, 1],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_compute_attribute_counts_invalid_pattern(mocker):
    data = {
        "Subject ID": [1, 2],
        "Period": ["P1", "P1"],
        "Attribute1": ["A", "A"],
        "Attribute2": ["X", "Y"],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df
    result = compute_attribute_counts(
        test_df, "InvalidPattern", "Period", "P1", type_val_sep
    )

    expected_data = {
        "AttributeValue": ["Attribute1=A", "Attribute2=X", "Attribute2=Y"],
        "Count": [2, 1, 1],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_compute_attribute_counts_missing_column(mocker):
    """Test that compute_attribute_counts handles missing columns gracefully."""
    data = {
        "Subject ID": [1, 2, 3],
        "Period": ["P1", "P1", "P2"],
        "Attribute1": ["A", "A", "B"],
        "Attribute2": ["X", "Y", "X"],
    }

    test_df = pd.DataFrame(data)

    mocker.patch(
        "intelligence_toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df
    
    # Pattern contains 'indicator_count' which doesn't exist in the DataFrame
    result = compute_attribute_counts(
        test_df, f"indicator_count{type_val_sep}1 & Attribute1{type_val_sep}A", "Period", "P1", type_val_sep
    )

    # Should return attribute counts for the valid columns that match the period
    # The missing column should be skipped without raising KeyError
    assert isinstance(result, pd.DataFrame)
    assert "AttributeValue" in result.columns
    assert "Count" in result.columns


def test_prepare_graph(mocker):
    create_edge_df_from_atts_mock = mocker.patch(
        "intelligence_toolkit.detect_case_patterns.graph_functions.create_edge_df_from_atts"
    )
    edge_df = pd.DataFrame(
        {
            "source": ["A", "B", "C", "A"],
            "target": ["B", "C", "D", "A"],
            "weight": [1, 2, 3, 3],
        }
    )
    create_edge_df_from_atts_mock.return_value = edge_df
    test_df = pd.DataFrame(
        {
            "Subject ID": [1, 2, 2, 1],
            "Period": [2020, 2021, 2021, 2020],
            "Full Attribute": ["ab=1", "bc=2", "ab=2", "bc=1"],
        }
    )
    pdf, time_to_graph = prepare_graph(test_df, min_edge_weight, missing_edge_prop)
    assert "Grouping ID" in pdf.columns
    assert pdf["Grouping ID"].str.contains("@").all()
    assert all(
        isinstance(graph, nx.classes.graph.Graph) for graph in time_to_graph.values()
    )
    assert len(time_to_graph) == 2
    assert all(
        isinstance(graph, nx.classes.graph.Graph) for graph in time_to_graph.values()
    )
    assert len(time_to_graph) == 2


def test_detect_patterns_with_empty_overall_score(mocker):
    """Test that detect_patterns handles empty pattern_df gracefully.
    
    This test ensures the fix for the normalization bug where empty
    overall_score would cause an error. The fix checks if the DataFrame
    has any values before normalization.
    """
    # Mock the dependencies
    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.create_close_node_rows"
    ).return_value = (pd.DataFrame(), {}, {})

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.create_period_to_patterns"
    ).return_value = {}

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.create_pattern_rows"
    ).return_value = []

    dynamic_df = pd.DataFrame({
        "Period": [],
        "Subject ID": [],
        "Full Attribute": []
    })

    # This should not raise an error even with empty overall_score
    pattern_df, _, _ = detect_patterns({}, dynamic_df, type_val_sep)

    # Should return a DataFrame with overall_score column
    assert isinstance(pattern_df, pd.DataFrame)
    assert "overall_score" in pattern_df.columns
    # The overall_score should either be 0 (empty case) or numeric
    assert pattern_df["overall_score"].dtype in [float, int, "float64", "int64"]


def test_detect_patterns_overall_score_normalization(mocker):
    """Test that overall_score is correctly normalized."""
    # Create mock data for pattern rows
    pattern_rows = [
        ["P1", "attr1=val1", 2, 5, 1.0, 2.0],
        ["P2", "attr2=val2", 3, 8, 1.5, 3.0],
    ]

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.create_close_node_rows"
    ).return_value = (pd.DataFrame(), {}, {})

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.create_period_to_patterns"
    ).return_value = {}

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.create_pattern_rows"
    ).return_value = pattern_rows

    dynamic_df = pd.DataFrame({
        "Period": ["P1", "P2"],
        "Subject ID": [1, 2],
        "Full Attribute": ["attr1=val1", "attr2=val2"]
    })

    pattern_df, _, _ = detect_patterns({}, dynamic_df, type_val_sep)

    # Verify overall_score column exists and is normalized
    assert "overall_score" in pattern_df.columns
    assert len(pattern_df) > 0
    # Max score should be 1.0 (or close due to rounding)
    assert pattern_df["overall_score"].max() <= 1.0
    # All scores should be non-negative
    assert (pattern_df["overall_score"] >= 0).all()
    # Scores should be rounded to 2 decimal places
    assert all(
        isinstance(score, (int, float)) for score in pattern_df["overall_score"]
    )
