# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
import plotly.graph_objs as go
from intelligence_toolkit.anonymize_case_data.visuals import (
    hsl_to_hex,
    hex_to_rgb,
    color_to_hsl,
    get_bar_chart,
    get_line_chart,
    get_flow_chart,
    print_selections,
    color_schemes,
)


def test_hsl_to_hex_red():
    # Pure red: H=0, S=100, L=50
    hex_color = hsl_to_hex(0, 100, 50)
    assert hex_color == "#ff0000"


def test_hsl_to_hex_blue():
    # Pure blue: H=240, S=100, L=50
    hex_color = hsl_to_hex(240, 100, 50)
    assert hex_color == "#0000ff"


def test_hsl_to_hex_green():
    # Pure green: H=120, S=100, L=50
    hex_color = hsl_to_hex(120, 100, 50)
    assert hex_color == "#00ff00"


def test_hsl_to_hex_gray():
    # Gray: S=0, any H, L=50
    hex_color = hsl_to_hex(0, 0, 50)
    assert hex_color.startswith("#")
    assert len(hex_color) == 7


def test_hex_to_rgb_red():
    rgb = hex_to_rgb("#ff0000")
    assert rgb == (255, 0, 0)


def test_hex_to_rgb_blue():
    rgb = hex_to_rgb("#0000ff")
    assert rgb == (0, 0, 255)


def test_hex_to_rgb_without_hash():
    rgb = hex_to_rgb("00ff00")
    assert rgb == (0, 255, 0)


def test_hex_to_rgb_mixed():
    rgb = hex_to_rgb("#a1b2c3")
    assert rgb == (161, 178, 195)


def test_color_to_hsl_from_hex():
    hsl = color_to_hsl("#ff0000")
    assert hsl[0] == 0  # Hue for red
    assert hsl[1] == 100  # Saturation
    assert hsl[2] == 50  # Lightness


def test_color_to_hsl_from_rgb():
    hsl = color_to_hsl("rgb(255,0,0)")
    assert hsl[0] == 0  # Hue for red
    assert hsl[1] == 100  # Saturation
    assert hsl[2] == 50  # Lightness


def test_color_to_hsl_blue():
    hsl = color_to_hsl("#0000ff")
    assert hsl[0] == 240  # Hue for blue


def test_get_bar_chart_basic():
    chart_df = pd.DataFrame(
        {
            "Attribute": ["Color", "Color", "Size"],
            "Attribute Value": ["Red", "Blue", "Large"],
            "Count": [10, 5, 8],
        }
    )
    selection = []
    show_attributes = ["Color"]
    unit = "record"
    width = 800
    height = 600
    scheme = color_schemes["Plotly"]

    fig = get_bar_chart(selection, show_attributes, unit, chart_df, width, height, scheme)

    assert isinstance(fig, go.Figure)
    assert fig.layout.width == width
    assert fig.layout.height == height


def test_get_bar_chart_with_selection():
    chart_df = pd.DataFrame(
        {
            "Attribute": ["Color"],
            "Attribute Value": ["Red"],
            "Count": [10],
        }
    )
    selection = [{"attribute": "Size", "value": "Large"}]
    show_attributes = []
    unit = "item"
    width = 600
    height = 400
    scheme = color_schemes["D3"]

    fig = get_bar_chart(selection, show_attributes, unit, chart_df, width, height, scheme)

    assert isinstance(fig, go.Figure)
    assert "matching" in fig.layout.title.text.lower()


def test_get_bar_chart_empty_unit():
    chart_df = pd.DataFrame(
        {
            "Attribute": ["Color"],
            "Attribute Value": ["Red"],
            "Count": [10],
        }
    )
    selection = []
    show_attributes = []
    unit = ""
    width = 800
    height = 600
    scheme = color_schemes["Plotly"]

    fig = get_bar_chart(selection, show_attributes, unit, chart_df, width, height, scheme)

    assert isinstance(fig, go.Figure)


def test_get_line_chart_basic():
    chart_df = pd.DataFrame(
        {
            "Year": ["2020", "2020", "2021"],
            "Attribute Value": ["Red", "Blue", "Red"],
            "Count": [10, 5, 12],
        }
    )
    selection = []
    series_attributes = ["Color"]
    unit = "record"
    time_attribute = "Year"
    width = 800
    height = 600
    scheme = color_schemes["Plotly"]

    fig = get_line_chart(
        selection, series_attributes, unit, chart_df, time_attribute, width, height, scheme
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.width == width
    assert fig.layout.height == height


def test_get_line_chart_with_selection():
    chart_df = pd.DataFrame(
        {
            "Year": ["2020"],
            "Attribute Value": ["Red"],
            "Count": [10],
        }
    )
    selection = [{"attribute": "Size", "value": "Large"}]
    series_attributes = ["Color"]
    unit = "case"
    time_attribute = "Year"
    width = 600
    height = 400
    scheme = color_schemes["Set1"]

    fig = get_line_chart(
        selection, series_attributes, unit, chart_df, time_attribute, width, height, scheme
    )

    assert isinstance(fig, go.Figure)
    assert "matching" in fig.layout.title.text.lower()


def test_get_flow_chart_basic():
    links_df = pd.DataFrame(
        {
            "Source": ["A", "B"],
            "Target": ["X", "Y"],
            "Count": [10, 5],
            "Highlight": [2, 1],
            "Proportion": [0.2, 0.2],
        }
    )
    selection = []
    source_attribute = "From"
    target_attribute = "To"
    highlight_attribute = ""
    width = 800
    height = 600
    unit = "record"
    scheme = color_schemes["Plotly"]

    fig = get_flow_chart(
        links_df,
        selection,
        source_attribute,
        target_attribute,
        highlight_attribute,
        width,
        height,
        unit,
        scheme,
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.width == width
    assert fig.layout.height == height


def test_get_flow_chart_with_highlight():
    links_df = pd.DataFrame(
        {
            "Source": ["A"],
            "Target": ["X"],
            "Count": [10],
            "Highlight": [3],
            "Proportion": [0.3],
        }
    )
    selection = []
    source_attribute = "From"
    target_attribute = "To"
    highlight_attribute = "Status:Active"
    width = 1000
    height = 700
    unit = "case"
    scheme = color_schemes["D3"]

    fig = get_flow_chart(
        links_df,
        selection,
        source_attribute,
        target_attribute,
        highlight_attribute,
        width,
        height,
        unit,
        scheme,
    )

    assert isinstance(fig, go.Figure)
    assert "colored by proportion" in fig.layout.title.text.lower()


def test_get_flow_chart_with_selection():
    links_df = pd.DataFrame(
        {
            "Source": ["A"],
            "Target": ["X"],
            "Count": [10],
            "Highlight": [0],
            "Proportion": [0.0],
        }
    )
    selection = [{"attribute": "Color", "value": "Red"}]
    source_attribute = "From"
    target_attribute = "To"
    highlight_attribute = ""
    width = 800
    height = 600
    unit = ""
    scheme = color_schemes["Plotly"]

    fig = get_flow_chart(
        links_df,
        selection,
        source_attribute,
        target_attribute,
        highlight_attribute,
        width,
        height,
        unit,
        scheme,
    )

    assert isinstance(fig, go.Figure)
    assert "matching" in fig.layout.title.text.lower()


def test_print_selections_single_attribute():
    selection = [
        {"attribute": "Color", "value": "Red"},
        {"attribute": "Color", "value": "Blue"},
    ]

    result = print_selections(selection, multiline=False)

    assert "Color:" in result
    assert "Red" in result
    assert "Blue" in result


def test_print_selections_multiple_attributes():
    selection = [
        {"attribute": "Color", "value": "Red"},
        {"attribute": "Size", "value": "Large"},
    ]

    result = print_selections(selection, multiline=False)

    assert "Color:" in result
    assert "Size:" in result
    assert "Red" in result
    assert "Large" in result


def test_print_selections_multiline():
    selection = [
        {"attribute": "Color", "value": "Red"},
        {"attribute": "Size", "value": "Large"},
    ]

    result = print_selections(selection, multiline=True)

    assert "- " in result
    assert "\n" in result


def test_print_selections_sorted():
    selection = [
        {"attribute": "Color", "value": "Red"},
        {"attribute": "Color", "value": "Blue"},
    ]

    result = print_selections(selection, multiline=False)

    # Values within same attribute should be sorted
    assert "Blue|Red" in result or "Color:Blue|Red" in result


def test_color_schemes_available():
    # Test that color schemes are defined
    assert "Plotly" in color_schemes
    assert "D3" in color_schemes
    assert "Set1" in color_schemes
    assert len(color_schemes) > 0


def test_color_schemes_contain_colors():
    # Test that schemes contain actual color values
    for scheme_name, colors in color_schemes.items():
        assert len(colors) > 0
        assert isinstance(colors, list)
