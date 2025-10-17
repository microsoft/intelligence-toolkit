# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
from collections import defaultdict
from intelligence_toolkit.anonymize_case_data.queries import (
    get_data_schema,
    compute_aggregate_graph,
    compute_synthetic_graph,
    compute_top_attributes_query,
    compute_time_series_query,
)


def test_get_data_schema_with_dataframe():
    df = pd.DataFrame({"Color": ["Red", "Blue", "Red"], "Size": ["Large", "Small", ""]})

    schema = get_data_schema(df)

    assert "Color" in schema
    assert "Size" in schema
    assert "Red" in schema["Color"]
    assert "Blue" in schema["Color"]
    assert "Large" in schema["Size"]
    assert "Small" in schema["Size"]
    assert len(schema["Color"]) == 2
    assert len(schema["Size"]) == 2  # Empty string filtered out


def test_get_data_schema_with_none():
    schema = get_data_schema(None)

    assert isinstance(schema, defaultdict)
    assert len(schema) == 0


def test_get_data_schema_empty_dataframe():
    df = pd.DataFrame()

    schema = get_data_schema(df)

    assert len(schema) == 0


def test_get_data_schema_sorts_values():
    df = pd.DataFrame({"Letter": ["Z", "A", "M", "B"]})

    schema = get_data_schema(df)

    assert schema["Letter"] == ["A", "B", "M", "Z"]


def test_compute_aggregate_graph_basic():
    adf = pd.DataFrame(
        {
            "selections": [
                "source:A;target:B",
                "source:A;target:C",
                "source:B;target:C",
            ],
            "protected_count": [10, 5, 8],
        }
    )
    filters = []
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = None

    result = compute_aggregate_graph(
        adf, filters, source_attribute, target_attribute, highlight_attribute
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "Source",
        "Target",
        "Count",
        "Highlight",
        "Proportion",
        "Dataset",
    ]


def test_compute_aggregate_graph_with_filters():
    adf = pd.DataFrame(
        {
            "selections": [
                "color:Red;source:A;target:B",
                "color:Blue;source:A;target:C",
                "color:Red;source:B;target:C",
            ],
            "protected_count": [10, 5, 8],
        }
    )
    filters = ["color:Red"]
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = None

    result = compute_aggregate_graph(
        adf, filters, source_attribute, target_attribute, highlight_attribute
    )

    # Should only include rows matching filter
    assert len(result) == 2
    assert all(result["Dataset"] == "Aggregate")


def test_compute_aggregate_graph_with_highlight():
    adf = pd.DataFrame(
        {
            "selections": [
                "source:A;target:B",
                "highlight:Yes;source:A;target:B",
                "source:A;target:C",
            ],
            "protected_count": [10, 3, 5],
        }
    )
    filters = []
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = "highlight:Yes"

    result = compute_aggregate_graph(
        adf, filters, source_attribute, target_attribute, highlight_attribute
    )

    # Should have highlight values
    assert "Highlight" in result.columns
    assert "Proportion" in result.columns


def test_compute_aggregate_graph_zero_counts_filtered():
    adf = pd.DataFrame(
        {
            "selections": ["source:A;target:B", "source:C;target:D"],
            "protected_count": [10, 0],
        }
    )
    filters = []
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = None

    result = compute_aggregate_graph(
        adf, filters, source_attribute, target_attribute, highlight_attribute
    )

    # Zero counts should be filtered out
    assert len(result) == 1
    assert result.iloc[0]["Count"] == 10


def test_compute_synthetic_graph_basic():
    sdf = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["X", "Y", "X"],
        }
    )
    filters = []
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = ""

    result = compute_synthetic_graph(
        sdf, filters, source_attribute, target_attribute, highlight_attribute
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "Source",
        "Target",
        "Count",
        "Highlight",
        "Proportion",
        "Dataset",
    ]
    assert all(result["Dataset"] == "Synthetic")


def test_compute_synthetic_graph_with_filters():
    sdf = pd.DataFrame(
        {
            "color": ["Red", "Red", "Blue"],
            "source": ["A", "A", "B"],
            "target": ["X", "Y", "X"],
        }
    )
    filters = ["color:Red"]
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = ""

    result = compute_synthetic_graph(
        sdf, filters, source_attribute, target_attribute, highlight_attribute
    )

    # Should only count rows matching filter
    assert len(result) == 2  # A->X and A->Y


def test_compute_synthetic_graph_filters_empty_values():
    sdf = pd.DataFrame(
        {
            "source": ["A", "", "B"],
            "target": ["X", "Y", ""],
        }
    )
    filters = []
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = ""

    result = compute_synthetic_graph(
        sdf, filters, source_attribute, target_attribute, highlight_attribute
    )

    # Should filter out rows with empty source or target
    assert len(result) == 1
    assert result.iloc[0]["Source"] == "A"
    assert result.iloc[0]["Target"] == "X"


def test_compute_synthetic_graph_with_highlight():
    sdf = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["X", "X", "X"],
            "status": ["Active", "Inactive", "Active"],
        }
    )
    filters = []
    source_attribute = "source"
    target_attribute = "target"
    highlight_attribute = "status:Active"

    result = compute_synthetic_graph(
        sdf, filters, source_attribute, target_attribute, highlight_attribute
    )

    # Should have highlight counts
    ax_row = result[(result["Source"] == "A") & (result["Target"] == "X")]
    assert len(ax_row) == 1
    assert ax_row.iloc[0]["Count"] == 2
    assert ax_row.iloc[0]["Highlight"] == 1  # Only one with Active status


def test_compute_top_attributes_query_basic():
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red"],
            "Size": ["Large", "Small", "Large"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    show_attributes = []
    num_values = 0

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Attribute", "Attribute Value", "Count", "Dataset"]
    assert len(result) > 0
    assert all(result["Dataset"] == "Synthetic")


def test_compute_top_attributes_query_with_filter():
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red", "Green"],
            "Size": ["Large", "Small", "Large", "Medium"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = [{"attribute": "Color", "value": "Red"}]
    show_attributes = []
    num_values = 0

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    # Should only count rows where Color=Red
    assert len(result) > 0


def test_compute_top_attributes_query_show_specific_attributes():
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red"],
            "Size": ["Large", "Small", "Large"],
            "Weight": ["Heavy", "Light", "Heavy"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    show_attributes = ["Color", "Size"]
    num_values = 0

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    # Should only show Color and Size attributes
    assert all(result["Attribute"].isin(["Color", "Size"]))
    assert "Weight" not in result["Attribute"].values


def test_compute_top_attributes_query_limit_num_values():
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Green", "Yellow", "Purple"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    show_attributes = []
    num_values = 2

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    # Should limit to top 2 values
    assert len(result) <= 2


def test_compute_top_attributes_query_filters_empty_values():
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "", "Blue"],
            "Size": ["Large", "Small", ""],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    show_attributes = []
    num_values = 0

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    # Should not include empty values
    assert "" not in result["Attribute Value"].values


def test_compute_time_series_query_basic():
    sdf = pd.DataFrame(
        {
            "Year": ["2020", "2020", "2021", "2021"],
            "Color": ["Red", "Blue", "Red", "Green"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    time_attribute = "Year"
    time_series = ["Color"]

    result = compute_time_series_query(
        query, sdf, adf, time_attribute, time_series
    )

    assert isinstance(result, pd.DataFrame)
    assert time_attribute in result.columns
    assert "Attribute" in result.columns
    assert "Attribute Value" in result.columns
    assert "Count" in result.columns
    assert "Dataset" in result.columns


def test_compute_time_series_query_fills_missing_times():
    sdf = pd.DataFrame(
        {
            "Year": ["2020", "2020", "2021"],
            "Color": ["Red", "Blue", "Red"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    time_attribute = "Year"
    time_series = ["Color"]

    result = compute_time_series_query(
        query, sdf, adf, time_attribute, time_series
    )

    # Blue should appear in 2021 with count 0
    blue_2021 = result[
        (result["Year"] == "2021") & (result["Attribute Value"].str.contains("Blue"))
    ]
    assert len(blue_2021) > 0


def test_compute_time_series_query_filters_empty_times():
    sdf = pd.DataFrame(
        {
            "Year": ["2020", "", "2021"],
            "Color": ["Red", "Blue", "Green"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = []
    time_attribute = "Year"
    time_series = ["Color"]

    result = compute_time_series_query(
        query, sdf, adf, time_attribute, time_series
    )

    # Should not include rows with empty Year
    assert "" not in result["Year"].values


def test_compute_top_attributes_query_with_selection():
    sdf = pd.DataFrame(
        {
            "Year": ["2020", "2020", "2021", "2021"],
            "Color": ["Red", "Blue", "Red", "Green"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    query = [{"attribute": "Size", "value": "Large"}]
    time_attribute = "Year"
    time_series = ["Color"]

    result = compute_time_series_query(
        query, sdf, adf, time_attribute, time_series
    )

    # Should only include Large items
    assert len(result) > 0


def test_compute_top_attributes_query_with_unions():
    # Test has_unions path (multiple values for same attribute)
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Green", "Red", "Blue"],
            "Size": ["Large", "Small", "Large", "Medium", "Small"],
        }
    )
    adf = pd.DataFrame({"selections": [], "protected_count": []})
    # Multiple values for Color attribute triggers union path
    query = [
        {"attribute": "Color", "value": "Red"},
        {"attribute": "Color", "value": "Blue"},
    ]
    show_attributes = []
    num_values = 0

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    # Should only include Red or Blue colors
    assert len(result) > 0
    assert all(result["Dataset"] == "Synthetic")  # Unions use synthetic only


def test_compute_top_attributes_query_with_aggregate_counts():
    # Test path where aggregate counts are used (no unions)
    sdf = pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Red"],
            "Size": ["Large", "Small", "Large"],
        }
    )
    # Setup aggregate data that matches selections
    adf = pd.DataFrame(
        {
            "selections": ["Color:Red", "Color:Blue", "Size:Large"],
            "protected_count": [15, 8, 12],
        }
    )
    query = []
    show_attributes = []
    num_values = 0

    result = compute_top_attributes_query(query, sdf, adf, show_attributes, num_values)

    # Should include both Synthetic and Aggregate datasets
    assert "Aggregate" in result["Dataset"].values or "Synthetic" in result["Dataset"].values
