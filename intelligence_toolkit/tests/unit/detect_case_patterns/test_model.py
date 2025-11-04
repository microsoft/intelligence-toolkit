# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import pandas as pd
import pytest

from intelligence_toolkit.detect_case_patterns.config import (
    min_edge_weight,
    missing_edge_prop,
    type_val_sep,
)
from intelligence_toolkit.detect_case_patterns.model import (
    compute_attribute_counts,
    create_time_series_df,
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

    # This should not raise an error even with empty patterns
    pattern_df, _, _ = detect_patterns({}, dynamic_df, type_val_sep)

    # Should return an empty DataFrame with base columns
    assert isinstance(pattern_df, pd.DataFrame)
    assert len(pattern_df) == 0
    assert list(pattern_df.columns) == ["period", "pattern", "length", "count", "mean", "z_score"]


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


def test_create_time_series_df_with_valid_patterns(mocker):
    """Test create_time_series_df with valid string patterns."""
    # Create a mock RecordCounter
    mock_record_counter = mocker.MagicMock()
    mock_record_counter.create_time_series_rows.side_effect = [
        [["2023-01", "attr1 & attr2", 5]],
        [["2023-02", "attr3 & attr4", 3]],
        [["2023-03", "attr5 & attr6 & attr7", 8]],
    ]

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.RecordCounter",
        return_value=mock_record_counter,
    )

    # Create pattern_df with valid string patterns
    pattern_df = pd.DataFrame({
        "period": ["2023-01", "2023-02", "2023-03"],
        "pattern": ["attr1 & attr2", "attr3 & attr4", "attr5 & attr6 & attr7"],
        "length": [2, 2, 3],
        "count": [5, 3, 8],
    })

    mock_model = mocker.MagicMock()

    # Act
    result = create_time_series_df(mock_model, pattern_df)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "pattern", "count"]
    assert len(result) == 3
    # Verify mock was called 3 times (once for each pattern)
    assert mock_record_counter.create_time_series_rows.call_count == 3


def test_create_time_series_df_with_nan_pattern_skips_gracefully(mocker):
    """Test that create_time_series_df handles NaN patterns gracefully by skipping them."""
    # Create pattern_df with NaN in pattern column
    pattern_df = pd.DataFrame({
        "period": ["2023-01", "2023-02", "2023-03"],
        "pattern": ["attr1 & attr2", float("nan"), "attr3 & attr4"],  # Middle pattern is NaN
        "length": [2, 1, 2],
        "count": [5, 3, 7],
    })

    mock_record_counter = mocker.MagicMock()
    mock_record_counter.create_time_series_rows.side_effect = [
        [["2023-01", "attr1 & attr2", 5]],
        [["2023-03", "attr3 & attr4", 7]],
    ]

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.RecordCounter",
        return_value=mock_record_counter,
    )

    mock_model = mocker.MagicMock()

    # Act
    result = create_time_series_df(mock_model, pattern_df)

    # Assert
    # Should successfully process valid patterns and skip NaN
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "pattern", "count"]
    assert len(result) == 2  # Only 2 valid patterns, NaN skipped
    assert result["pattern"].tolist() == ["attr1 & attr2", "attr3 & attr4"]


def test_create_time_series_df_with_none_pattern_skips_gracefully(mocker):
    """Test that create_time_series_df handles None patterns gracefully by skipping them."""
    # Create pattern_df with None in pattern column
    pattern_df = pd.DataFrame({
        "period": ["2023-01", "2023-02", "2023-03"],
        "pattern": ["attr1 & attr2", None, "attr3 & attr4"],
        "length": [2, 1, 2],
        "count": [5, 3, 7],
    })

    mock_record_counter = mocker.MagicMock()
    mock_record_counter.create_time_series_rows.side_effect = [
        [["2023-01", "attr1 & attr2", 5]],
        [["2023-03", "attr3 & attr4", 7]],
    ]

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.RecordCounter",
        return_value=mock_record_counter,
    )

    mock_model = mocker.MagicMock()

    # Act
    result = create_time_series_df(mock_model, pattern_df)

    # Assert
    # Should successfully process valid patterns and skip None
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "pattern", "count"]
    assert len(result) == 2  # Only 2 valid patterns, None skipped
    assert result["pattern"].tolist() == ["attr1 & attr2", "attr3 & attr4"]


def test_create_time_series_df_with_numeric_pattern_skips_gracefully(mocker):
    """Test that create_time_series_df handles numeric patterns gracefully by skipping them."""
    # Create pattern_df with numeric value in pattern column
    pattern_df = pd.DataFrame({
        "period": ["2023-01", "2023-02", "2023-03"],
        "pattern": ["attr1 & attr2", 123, "attr3 & attr4"],  # Numeric value instead of string
        "length": [2, 1, 2],
        "count": [5, 3, 7],
    })

    mock_record_counter = mocker.MagicMock()
    mock_record_counter.create_time_series_rows.side_effect = [
        [["2023-01", "attr1 & attr2", 5]],
        [["2023-03", "attr3 & attr4", 7]],
    ]

    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.RecordCounter",
        return_value=mock_record_counter,
    )

    mock_model = mocker.MagicMock()

    # Act
    result = create_time_series_df(mock_model, pattern_df)

    # Assert
    # Should successfully process valid patterns and skip numeric ones
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "pattern", "count"]
    assert len(result) == 2  # Only 2 valid patterns, numeric skipped
    assert result["pattern"].tolist() == ["attr1 & attr2", "attr3 & attr4"]


def test_create_time_series_df_with_empty_pattern_df(mocker):
    """Test create_time_series_df with an empty pattern_df."""
    # Create empty pattern_df
    pattern_df = pd.DataFrame({
        "period": [],
        "pattern": [],
        "length": [],
        "count": [],
    })

    mock_record_counter = mocker.MagicMock()
    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.RecordCounter",
        return_value=mock_record_counter,
    )

    mock_model = mocker.MagicMock()

    # Act
    result = create_time_series_df(mock_model, pattern_df)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "pattern", "count"]
    assert len(result) == 0


def test_create_time_series_df_with_nan_pattern(mocker):
    """Test create_time_series_df with NaN pattern values (no converging patterns found)."""
    # Create pattern_df with NaN values (result of no converging pairs)
    pattern_df = pd.DataFrame({
        "period": [float("nan")],
        "pattern": [float("nan")],
        "length": [float("nan")],
        "count": [float("nan")],
        "mean": [float("nan")],
        "z_score": [float("nan")],
        "detections": [float("nan")],
        "overall_score": [0],
    })

    mock_record_counter = mocker.MagicMock()
    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.model.RecordCounter",
        return_value=mock_record_counter,
    )

    mock_model = mocker.MagicMock()

    # Act
    result = create_time_series_df(mock_model, pattern_df)

    # Assert - should return empty DataFrame with proper columns
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "pattern", "count"]
    assert len(result) == 0
    assert mock_record_counter.create_time_series_rows.call_count == 0


def test_detect_patterns_with_no_convergence(mocker):
    """Test detect_patterns when no converging pairs exist (close_node_df is empty)."""
    # Create minimal graph model with identical topology across periods
    model_data = {
        "Subject ID": ["1", "2", "3"],
        "Period": ["2020-Q1", "2020-Q1", "2020-Q1"],
        "Full Attribute": ["attr:val1", "attr:val2", "attr:val3"],
        "Attribute Type": ["attr", "attr", "attr"],
    }
    model = pd.DataFrame(model_data)

    # Create mock node_to_period_to_pos with embeddings that never converge
    # (all nodes at same position in every period)
    node_to_period_to_pos = {
        "attr:val1": {"ALL": [0.0, 0.0], "2020-Q1": [0.0, 0.0]},
        "attr:val2": {"ALL": [1.0, 1.0], "2020-Q1": [1.0, 1.0]},
        "attr:val3": {"ALL": [2.0, 2.0], "2020-Q1": [2.0, 2.0]},
    }

    # Mock is_converging_pair to always return False (no convergence)
    mocker.patch(
        "intelligence_toolkit.detect_case_patterns.detection_functions.is_converging_pair",
        return_value=False,
    )

    # Act
    patterns_df, close_pairs, all_pairs = detect_patterns(
        node_to_period_to_pos,
        model,
        type_val_sep,
        min_pattern_count=1,
        max_pattern_length=10,
    )

    # Assert
    assert close_pairs == 0
    assert isinstance(patterns_df, pd.DataFrame)
    assert len(patterns_df) == 0
    # Empty DataFrame has base columns only
    assert list(patterns_df.columns) == [
        "period",
        "pattern",
        "length",
        "count",
        "mean",
        "z_score",
    ]


