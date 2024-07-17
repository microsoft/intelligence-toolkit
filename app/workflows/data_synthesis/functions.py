# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import colorsys
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import workflows.data_synthesis.config as config


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


def flow_chart(
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
    default_color = config.color_schemes[scheme][0]
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


def compute_aggregate_graph(
    adf, filters, source_attribute, target_attribute, highlight_attribute
):
    edge_atts = {source_attribute, target_attribute}
    edges = []
    edge_counts = {}
    edge_highlights = {}
    for i, row in adf.iterrows():
        selections = row["selections"].split(config.att_separator)
        selection_atts = set([x.split(config.val_separator)[0] for x in selections])
        att_intersection = selection_atts.intersection(edge_atts)
        if att_intersection == edge_atts:
            # the record has both edge attributes; now we need to filter on the filters
            val_intersection = set(selections).intersection(filters)
            if val_intersection == set(filters):
                # the record has both edge attributes and the filters
                # check what else is in the selections
                remaining = []
                source_val = None
                target_val = None
                for selection in selections:
                    att = selection.split(config.val_separator)[0]
                    if att == source_attribute:
                        source_val = selection.split(config.val_separator)[1]
                    if att == target_attribute:
                        target_val = selection.split(config.val_separator)[1]
                    if (
                        att not in edge_atts
                        and selection not in filters
                        and selection not in remaining
                    ):
                        remaining.append(selection)
                if len(remaining) == 0:
                    edge_counts[(source_val, target_val)] = row["protected_count"]
                elif (
                    len(remaining) == 1
                    and highlight_attribute is not None
                    and remaining[0] == highlight_attribute
                ):
                    edge_highlights[(source_val, target_val)] = row["protected_count"]

    for edge, count in edge_counts.items():
        if count > 0:
            highlight = edge_highlights[edge] if edge in edge_highlights else 0
            edges.append(
                [
                    edge[0],
                    edge[1],
                    count,
                    highlight,
                    highlight / count,
                    "Aggregate",
                ]
            )

    edges_df = pd.DataFrame(
        edges,
        columns=["Source", "Target", "Count", "Highlight", "Proportion", "Dataset"],
    )
    return edges_df


def compute_synthetic_graph(
    sdf, filters, source_attribute, target_attribute, highlight_attribute
):
    edges = []
    att_groups = {}
    for f in filters:
        att, val = f.split(config.val_separator)
        att_groups[att] = att_groups.get(att, []) + [val]

    # compute all pairs of source and target
    for source in sdf[source_attribute].unique():
        for target in sdf[target_attribute].unique():
            if len(str(source)) == 0 or len(str(target)) == 0:
                continue
            df = sdf.copy()
            df = df[df[source_attribute] == source]
            df = df[df[target_attribute] == target]
            for att, vals in att_groups.items():
                df = df[df[att].isin(vals)]
            count = len(df)
            if count > 0:
                highlight = 0
                if highlight_attribute != "":
                    hatt, hval = highlight_attribute.split(config.val_separator)
                    df = df[df[hatt] == hval]
                    highlight = len(df)
                edges.append(
                    [
                        source,
                        target,
                        count,
                        highlight,
                        highlight / count,
                        "Synthetic",
                    ]
                )

    edges_df = pd.DataFrame(
        edges,
        columns=["Source", "Target", "Count", "Highlight", "Proportion", "Dataset"],
    )
    return edges_df


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
        color_discrete_sequence=config.color_schemes[scheme],
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
    title = f'Time series for {", ".join(series_attributes)} attributes across all {unit.title()} records'.replace(
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
        color_discrete_sequence=config.color_schemes[scheme],
    )
    fig.update_layout(
        yaxis={"title_text": f"{unit.title()} Count" if unit != "" else "Count"}
    )
    return fig


def compute_top_attributes_query(
    query,
    sdf,
    adf,
    att_separator,
    val_separator,
    data_schema,
    show_attributes,
    num_values,
):
    # print(f'compute_query with query {query}')
    df = sdf.copy(deep=True)
    selection = []
    has_unions = False
    for att, vals in data_schema.items():
        filter_vals = [
            v
            for v in vals
            if {"attribute": att, "value": v} in query and len(str(v)) > 0
        ]
        if len(filter_vals) > 0:
            df = df[df[att].isin(filter_vals)]
            if len(filter_vals) == 1:
                selection.append(f"{att}{val_separator}{filter_vals[0]}")
            else:
                has_unions = True
    # Add ID based on row number
    df["Id"] = [i for i in range(len(df))]
    sdf_filtered = df.melt(id_vars=["Id"], var_name="Attribute", value_name="Value")
    sdf_filtered["AttributeValue"] = (
        sdf_filtered["Attribute"] + val_separator + sdf_filtered["Value"]
    )
    sdf_filtered = sdf_filtered[sdf_filtered["Value"] != ""]
    syn_counts = (
        sdf_filtered["AttributeValue"]
        .value_counts()
        .rename_axis("Attribute Value")
        .to_frame("Count")
    )
    syn_counts["Attribute"] = syn_counts.index.str.split(val_separator).str[0]
    syn_counts.reset_index(level=0, inplace=True)
    syn_counts["Dataset"] = "Synthetic"

    result_df = syn_counts[["Attribute", "Attribute Value", "Count", "Dataset"]]

    if not has_unions:
        agg_rows = []
        for att, vals in data_schema.items():
            for val in vals:
                filter = f"{att}{val_separator}{val}"
                extended_selection = (
                    sorted(selection + [filter])
                    if filter not in selection
                    else sorted(selection)
                )
                extended_selection_key = att_separator.join(extended_selection)
                extended_filtered_aggs = adf[
                    adf["selections"] == extended_selection_key
                ]
                extended_agg_count = (
                    extended_filtered_aggs["protected_count"].values[0]
                    if len(extended_filtered_aggs) > 0
                    else 0
                )
                if extended_agg_count > 0:
                    agg_rows.append([att, filter, extended_agg_count, "Aggregate"])
        agg_df = pd.DataFrame(
            agg_rows, columns=["Attribute", "Attribute Value", "Count", "Dataset"]
        )
        result_df = pd.concat([syn_counts, agg_df], axis=0, ignore_index=True)
        # remove rows of result_df where Dataset = Synthetic if there is another row with Dataset = Aggregate for the same Attribute Value
        result_df = result_df[
            ~(
                (result_df["Dataset"] == "Synthetic")
                & (
                    result_df["Attribute Value"].isin(
                        result_df[result_df["Dataset"] == "Aggregate"][
                            "Attribute Value"
                        ].values
                    )
                )
            )
        ]
    if len(show_attributes) > 0:
        result_df = result_df[result_df["Attribute"].isin(show_attributes)]

    result_df = result_df.sort_values(by=["Count"], ascending=False)
    if num_values > 0:
        result_df = result_df[:num_values]
    return result_df[["Attribute", "Attribute Value", "Count", "Dataset"]]


def compute_time_series_query(
    query,
    sdf,
    adf,
    att_separator,
    val_separator,
    data_schema,
    time_attribute,
    time_series,
):
    tdfs = []
    times = [t for t in sorted(sdf[time_attribute].unique()) if len(str(t)) > 0]
    for time in times:
        time_query = query + [{"attribute": time_attribute, "value": time}]
        tdf = compute_top_attributes_query(
            time_query,
            sdf,
            adf,
            att_separator,
            val_separator,
            data_schema,
            time_series,
            0,
        )
        tdf[time_attribute] = time
        tdfs.append(tdf)
    final_tdf = pd.concat(tdfs, axis=0, ignore_index=True)
    missing = []
    for _, row in final_tdf.iterrows():
        att = row["Attribute"]
        val = row["Attribute Value"]
        for time in times:
            match = final_tdf[
                (final_tdf["Attribute"] == att)
                & (final_tdf["Attribute Value"] == val)
                & (final_tdf[time_attribute] == time)
            ]
            if len(match) == 0:
                missing.append([time, att, val, 0, "Aggregate"])

    if len(missing) > 0:
        missing_df = pd.DataFrame(
            missing,
            columns=[
                time_attribute,
                "Attribute",
                "Attribute Value",
                "Count",
                "Dataset",
            ],
        )
        final_tdf = pd.concat([final_tdf, missing_df], axis=0, ignore_index=True)
    return (
        final_tdf[[time_attribute, "Attribute", "Attribute Value", "Count", "Dataset"]]
        .drop_duplicates()
        .sort_values(by=[time_attribute, "Attribute Value"])
    )


def make_random_data_table(num_rows, num_cols):
    random_df = pd.DataFrame()
    for i in range(num_cols):
        if random.random() < 0.5:
            limit = random.randint(1, 1000)
            random_df[f"col{i}"] = [random.randint(0, limit) for _ in range(num_rows)]
        else:
            random_df[f"col{i}"] = [random.random() for _ in range(num_rows)]

    random_df.to_csv(os.path.join(config.outputs_dir, "random_data.csv"), index=False)
