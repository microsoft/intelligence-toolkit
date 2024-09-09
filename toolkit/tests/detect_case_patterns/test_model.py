# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import pandas as pd

from toolkit.detect_case_patterns.config import (
    min_edge_weight,
    missing_edge_prop,
    type_val_sep,
)
from toolkit.detect_case_patterns.model import (
    compute_attribute_counts,
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

    mocker.patch("toolkit.helpers.df_functions.fix_null_ints").return_value = test_df
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
        "toolkit.helpers.df_functions.fix_null_ints"
    ).return_value = test_df.fillna("")
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

    mocker.patch("toolkit.helpers.df_functions.fix_null_ints").return_value = test_df
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

    mocker.patch("toolkit.helpers.df_functions.fix_null_ints").return_value = test_df
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

    mocker.patch("toolkit.helpers.df_functions.fix_null_ints").return_value = test_df
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

    mocker.patch("toolkit.helpers.df_functions.fix_null_ints").return_value = test_df
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

    mocker.patch("toolkit.helpers.df_functions.fix_null_ints").return_value = test_df
    result = compute_attribute_counts(
        test_df, "InvalidPattern", "Period", "P1", type_val_sep
    )

    expected_data = {
        "AttributeValue": ["Attribute1=A", "Attribute2=X", "Attribute2=Y"],
        "Count": [2, 1, 1],
    }

    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


def test_prepare_graph(mocker):
    create_edge_df_from_atts_mock = mocker.patch(
        "toolkit.detect_case_patterns.graph_functions.create_edge_df_from_atts"
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
