# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
import math
from unittest.mock import MagicMock, patch, Mock
from intelligence_toolkit.anonymize_case_data.api import AnonymizeCaseData
from intelligence_toolkit.anonymize_case_data.synthesizability_statistics import (
    SynthesizabilityStatistics,
)


def test_anonymize_case_data_initialization():
    acd = AnonymizeCaseData()

    assert acd.protected_number_of_records == 0
    assert acd.delta == 0
    assert isinstance(acd.sensitive_df, pd.DataFrame)
    assert isinstance(acd.aggregate_df, pd.DataFrame)
    assert isinstance(acd.synthetic_aggregate_df, pd.DataFrame)
    assert isinstance(acd.synthetic_df, pd.DataFrame)
    assert isinstance(acd.aggregate_error_report, pd.DataFrame)
    assert isinstance(acd.synthetic_error_report, pd.DataFrame)


def test_fabrication_strategy_enum():
    # Test that enum values exist
    assert hasattr(AnonymizeCaseData.FabricationStrategy, "BALANCED")
    assert hasattr(AnonymizeCaseData.FabricationStrategy, "PROGRESSIVE")
    assert hasattr(AnonymizeCaseData.FabricationStrategy, "MINIMIZED")
    assert hasattr(AnonymizeCaseData.FabricationStrategy, "UNCONTROLLED")


def test_analyze_synthesizability_basic():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red"],
            "Size": ["Large", "Small", "Large"],
        }
    )

    stats = acd.analyze_synthesizability(df)

    assert isinstance(stats, SynthesizabilityStatistics)
    assert stats.num_cols == 2
    assert stats.overall_att_count > 0
    assert stats.possible_combinations > 0


def test_analyze_synthesizability_with_empty_values():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "Color": ["Red", "", "Blue"],
            "Size": ["Large", "Small", ""],
        }
    )

    stats = acd.analyze_synthesizability(df)

    # Empty values should be filtered out
    assert isinstance(stats, SynthesizabilityStatistics)
    assert stats.num_cols == 2


def test_analyze_synthesizability_with_nan():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "Color": ["Red", None, "Blue"],
            "Size": ["Large", "Small", None],
        }
    )

    stats = acd.analyze_synthesizability(df)

    # NaN values should be filtered out
    assert isinstance(stats, SynthesizabilityStatistics)


def test_analyze_synthesizability_distinct_counts():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "Color": ["Red", "Red", "Red"],
            "Size": ["Large", "Large", "Large"],
        }
    )

    stats = acd.analyze_synthesizability(df)

    # Only 1 distinct value per column
    assert stats.possible_combinations == 1  # 1 * 1


def test_analyze_synthesizability_calculates_combinations():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "Color": ["Red", "Blue"],
            "Size": ["Large", "Small"],
        }
    )

    stats = acd.analyze_synthesizability(df)

    # 2 colors * 2 sizes = 4 possible combinations
    assert stats.possible_combinations == 4
    assert stats.possible_combinations_per_row == 2.0  # 4 / 2 rows


def test_analyze_synthesizability_mean_vals_per_record():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "A": ["X", "Y"],
            "B": ["1", ""],
            "C": ["", "2"],
        }
    )

    stats = acd.analyze_synthesizability(df)

    # Row 1: 2 vals (X, 1), Row 2: 2 vals (Y, 2), mean = 2.0
    assert stats.mean_vals_per_record == 2.0


def test_analyze_synthesizability_excess_combinations_ratio():
    acd = AnonymizeCaseData()
    df = pd.DataFrame(
        {
            "A": ["X", "Y"],
            "B": ["1", "2"],
        }
    )

    stats = acd.analyze_synthesizability(df)

    # Should calculate excess_combinations_ratio
    assert stats.excess_combinations_ratio > 0


@patch("intelligence_toolkit.anonymize_case_data.api.DpAggregateSeededSynthesizer")
@patch("intelligence_toolkit.anonymize_case_data.api.Dataset")
@patch("intelligence_toolkit.anonymize_case_data.api.df_functions.fix_null_ints")
def test_anonymize_case_data_method(mock_fix_null_ints, mock_dataset, mock_synth_class):
    # Setup mocks
    mock_fix_null_ints.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    mock_dataset_instance = MagicMock()
    mock_dataset.from_data_frame.return_value = mock_dataset_instance
    mock_dataset_instance.get_aggregates.return_value = {"A:1": 10}
    mock_dataset.return_value.get_aggregates.return_value = {"A:1": 9}
    mock_dataset.raw_data_to_data_frame.return_value = pd.DataFrame({"A": [1], "B": [3]})

    mock_synth_instance = MagicMock()
    mock_synth_class.return_value = mock_synth_instance
    mock_synth_instance.get_dp_number_of_records.return_value = 100
    mock_synth_instance.get_dp_aggregates.return_value = {"A:1": 10}
    mock_synth_instance.sample.return_value = MagicMock()

    acd = AnonymizeCaseData()
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    acd.anonymize_case_data(df, epsilon=1.0)

    # Verify synthesizer was created and fit was called
    assert mock_synth_instance.fit.called
    assert mock_synth_instance.sample.called
    assert acd.protected_number_of_records == 100


def test_get_data_schema():
    acd = AnonymizeCaseData()
    acd.synthetic_df = pd.DataFrame({"Color": ["Red", "Blue"], "Size": ["Large", "Small"]})

    schema = acd.get_data_schema()

    assert "Color" in schema
    assert "Size" in schema


@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_aggregate_graph")
def test_compute_aggregate_graph_df(mock_compute):
    mock_compute.return_value = pd.DataFrame({"Source": ["A"], "Target": ["B"]})

    acd = AnonymizeCaseData()
    acd.aggregate_df = pd.DataFrame({"selections": ["A;B"], "protected_count": [10]})

    result = acd.compute_aggregate_graph_df([], "source", "target", "")

    assert mock_compute.called
    assert isinstance(result, pd.DataFrame)


@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_synthetic_graph")
def test_compute_synthetic_graph_df(mock_compute):
    mock_compute.return_value = pd.DataFrame({"Source": ["A"], "Target": ["B"]})

    acd = AnonymizeCaseData()
    acd.synthetic_df = pd.DataFrame({"source": ["A"], "target": ["B"]})

    result = acd.compute_synthetic_graph_df([], "source", "target", "")

    assert mock_compute.called
    assert isinstance(result, pd.DataFrame)


@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_time_series_query")
def test_compute_time_series_query_df(mock_compute):
    mock_compute.return_value = pd.DataFrame({"Year": ["2020"], "Count": [10]})

    acd = AnonymizeCaseData()
    acd.synthetic_df = pd.DataFrame({"Year": ["2020"], "Value": [10]})
    acd.aggregate_df = pd.DataFrame({"selections": [], "protected_count": []})

    result = acd.compute_time_series_query_df([], "Year", ["Value"])

    assert mock_compute.called
    assert isinstance(result, pd.DataFrame)


@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_top_attributes_query")
def test_compute_top_attributes_query_df(mock_compute):
    mock_compute.return_value = pd.DataFrame({"Attribute": ["Color"], "Count": [10]})

    acd = AnonymizeCaseData()
    acd.synthetic_df = pd.DataFrame({"Color": ["Red"]})
    acd.aggregate_df = pd.DataFrame({"selections": [], "protected_count": []})

    result = acd.compute_top_attributes_query_df([], ["Color"], 10)

    assert mock_compute.called
    assert isinstance(result, pd.DataFrame)


@patch("intelligence_toolkit.anonymize_case_data.api.visuals.get_bar_chart")
def test_get_bar_chart_fig(mock_get_chart):
    mock_fig = MagicMock()
    mock_get_chart.return_value = mock_fig

    acd = AnonymizeCaseData()
    acd.synthetic_df = pd.DataFrame({"Color": ["Red", "Blue"]})
    acd.aggregate_df = pd.DataFrame({"selections": [], "protected_count": []})

    fig, chart_df = acd.get_bar_chart_fig([], ["Color"], "record", 800, 600, ["#ff0000"], 10)

    assert mock_get_chart.called
    assert fig == mock_fig


@patch("intelligence_toolkit.anonymize_case_data.api.visuals.get_line_chart")
@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_time_series_query")
def test_get_line_chart_fig(mock_compute, mock_get_chart):
    mock_fig = MagicMock()
    mock_get_chart.return_value = mock_fig
    mock_compute.return_value = pd.DataFrame({"Year": ["2020"], "Count": [10]})

    acd = AnonymizeCaseData()
    acd.synthetic_df = pd.DataFrame({"Year": ["2020"], "Value": [10]})
    acd.aggregate_df = pd.DataFrame({"selections": [], "protected_count": []})

    fig, chart_df = acd.get_line_chart_fig([], ["Value"], "record", "Year", 800, 600, ["#ff0000"])

    assert mock_get_chart.called
    assert fig == mock_fig


@patch("intelligence_toolkit.anonymize_case_data.api.visuals.get_flow_chart")
@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_aggregate_graph")
def test_get_flow_chart_fig_with_aggregate(mock_compute_agg, mock_get_chart):
    mock_fig = MagicMock()
    mock_get_chart.return_value = mock_fig
    mock_compute_agg.return_value = pd.DataFrame({"Source": ["A"], "Target": ["B"]})

    acd = AnonymizeCaseData()
    acd.aggregate_df = pd.DataFrame({"selections": [], "protected_count": []})
    acd.synthetic_df = pd.DataFrame({"source": ["A"], "target": ["B"]})

    # With 2 attributes (source + target), should use aggregate
    fig, chart_df = acd.get_flow_chart_fig([], "source", "target", "", 800, 600, "record", ["#ff0000"])

    assert mock_compute_agg.called
    assert fig == mock_fig


@patch("intelligence_toolkit.anonymize_case_data.api.visuals.get_flow_chart")
@patch("intelligence_toolkit.anonymize_case_data.api.queries.compute_synthetic_graph")
def test_get_flow_chart_fig_with_synthetic(mock_compute_syn, mock_get_chart):
    mock_fig = MagicMock()
    mock_get_chart.return_value = mock_fig
    mock_compute_syn.return_value = pd.DataFrame({"Source": ["A"], "Target": ["B"]})

    acd = AnonymizeCaseData()
    acd.aggregate_df = pd.DataFrame({"selections": [], "protected_count": []})
    acd.synthetic_df = pd.DataFrame({"source": ["A"], "target": ["B"]})

    # With many selections (> 4 attributes), should use synthetic
    selection = [
        {"attribute": "A", "value": "1"},
        {"attribute": "B", "value": "2"},
        {"attribute": "C", "value": "3"},
    ]
    fig, chart_df = acd.get_flow_chart_fig(
        selection, "source", "target", "", 800, 600, "record", ["#ff0000"]
    )

    assert mock_compute_syn.called
    assert fig == mock_fig


def test_analyze_synthesizability_single_row():
    acd = AnonymizeCaseData()
    # Test with single row
    df = pd.DataFrame({"A": [1], "B": [2]})

    stats = acd.analyze_synthesizability(df)

    assert stats.num_cols == 2
    assert stats.possible_combinations_per_row == 1.0  # 1 combination / 1 row
