# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import colorsys
from collections import defaultdict
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

color_schemes = {
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "G10": px.colors.qualitative.G10,
    "T10": px.colors.qualitative.T10,
    "Alphabet": px.colors.qualitative.Alphabet,
    "Dark24": px.colors.qualitative.Dark24,
    "Light24": px.colors.qualitative.Light24,
    "Set1": px.colors.qualitative.Set1,
    "Pastel1": px.colors.qualitative.Pastel1,
    "Dark2": px.colors.qualitative.Dark2,
    "Set2": px.colors.qualitative.Set2,
    "Pastel2": px.colors.qualitative.Pastel2,
    "Set3": px.colors.qualitative.Set3,
    "Antique": px.colors.qualitative.Antique,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Prism": px.colors.qualitative.Prism,
    "Safe": px.colors.qualitative.Safe,
    "Vivid": px.colors.qualitative.Vivid,
}

def get_bar_chart(selection, show_attributes, unit, chart_df, width, height, scheme):
    title = f'Top {"" if len(show_attributes) == 0 else ", ".join(show_attributes)} attributes across all {unit.title()} records'.replace(
        "  ", " "
    )
    if len(selection) > 0:
        title += (
            ' matching "'.replace("  ", " ")
            + print_selections(selection, multiline=False)
            + '"'
        )
    chart_df = chart_df.copy(deep=True)
    fig = px.bar(
        chart_df,
        x="Attribute Value",
        y="Count",
        color="Attribute",
        orientation="v",
        text_auto=True,
        title=title,
        width=width,
        height=height,
        color_discrete_sequence=scheme,
    )
    fig.update_xaxes(title_text="")
    fig.update_layout(
        yaxis={
            "categoryorder": "total descending",
            "title_text": f"{unit.title()} Count" if unit != "" else "Count",
        }
    )
    return fig

def get_line_chart(
    selection, series_attributes, unit, chart_df, time_attribute, width, height, scheme
):
    title = f'Time series across all {unit.title()} records'.replace(
        "  ", " "
    )
    if len(selection) > 0:
        title += (
            ' matching "'.replace("  ", " ")
            + print_selections(selection, multiline=False)
            + '"'
        )
    chart_df = chart_df.copy(deep=True)
    fig = px.line(
        chart_df,
        x=time_attribute,
        y="Count",
        color="Attribute Value",
        orientation="v",
        title=title,
        width=width,
        height=height,
        color_discrete_sequence=scheme,
    )
    fig.update_layout(
        yaxis={"title_text": f"{unit.title()} Count" if unit != "" else "Count"}
    )
    return fig

def hsl_to_hex(h, s, l):
    rgb = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    rgb = tuple([int(x * 255) for x in rgb])
    hex_color = "#%02x%02x%02x" % rgb
    return hex_color


def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    rgb = tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))
    return rgb


def color_to_hsl(color):
    if color.startswith("rgb"):
        rgb = tuple([int(x) for x in color[4:-1].split(",")])
    else:
        rgb = hex_to_rgb(color)
    hls = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    return [int(hls[0] * 360), int(hls[2] * 100), int(hls[1] * 100)]


def get_flow_chart(
    links_df,
    selection,
    source_attribute,
    target_attribute,
    highlight_attribute,
    width,
    height,
    unit,
    scheme,
):
    title = f"{source_attribute}\u2014{target_attribute} links for all {unit.title()} records".replace(
        "  ", " "
    )
    if len(selection) > 0:
        title += (
            ' matching "'.replace("  ", " ")
            + print_selections(selection, multiline=False)
            + '"'
        )
    title += (
        f",<br>colored by proportion with {highlight_attribute}".replace("  ", " ")
        if highlight_attribute != ""
        else ""
    )
    if unit != "":
        unit = unit + " "
    if highlight_attribute != "":
        highlight_attribute = highlight_attribute + " "
    nodes = sorted(
        list(
            set(
                links_df["Source"].unique().tolist()
                + [x + " " for x in links_df["Target"].unique().tolist()]
            )
        )
    )
    default_color = scheme[0]
    h, s, l = color_to_hsl(default_color.lstrip("#"))
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=default_color,
                ),
                link=dict(
                    source=[
                        nodes.index(x) for x in links_df["Source"].tolist()
                    ],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=[nodes.index(x + " ") for x in links_df["Target"].tolist()],
                    value=links_df["Count"].tolist(),
                    color=[
                        hsl_to_hex(int(h), int(p * 100), 70)
                        for p in links_df["Proportion"].tolist()
                    ],
                    hovertemplate=source_attribute
                    + ": %{source.label} + "
                    + target_attribute
                    + ": %{target.label} = %{value:.0f}<br>+ "
                    + highlight_attribute
                    + " = %{customdata[0]}<br>Proportion = %{customdata[1]:.1%}<extra></extra>",
                    customdata=np.stack(
                        (
                            links_df["Highlight"].tolist(),
                            links_df["Proportion"].tolist(),
                        ),
                        axis=-1,
                    ),
                ),
            )
        ]
    )

    fig.update_layout(font_size=14, width=width, height=height, title_text=title)

    return fig


def print_selections(selection, multiline=True):
    sd = defaultdict(list)
    for x in selection:
        sd[x["attribute"]].append(x["value"])
    for k, vs in sd.items():
        vs.sort()
    ks = sorted(sd.keys())
    text = ""
    if multiline:
        for k, vs in sd.items():
            text += f"- {k} = " + " | ".join(vs) + "\n"
    else:
        text = ", ".join([f"{k}:" + "|".join(vs) for k, vs in sd.items()])
    return text
