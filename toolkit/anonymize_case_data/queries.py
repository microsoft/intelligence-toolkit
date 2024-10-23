# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

from collections import defaultdict
from typing import Any

import pandas as pd


def get_data_schema(sdf) -> dict[list[str]]:
    data_schema = defaultdict(list)
    if sdf is not None:
        for att in sdf.columns.values:
            vals = [str(x) for x in sdf[att].unique() if len(str(x)) > 0]
            for val in vals:
                data_schema[att].append(val)
            data_schema[att].sort()
    return data_schema

def compute_aggregate_graph(
    adf,
    filters,
    source_attribute,
    target_attribute,
    highlight_attribute,
    att_separator=";",
    val_separator=":",
) -> pd.DataFrame:
    edge_atts = {source_attribute, target_attribute}
    edges = []
    edge_counts = {}
    edge_highlights = {}
    for i, row in adf.iterrows():
        selections = row["selections"].split(att_separator)
        selection_atts = set([x.split(val_separator)[0] for x in selections])
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
                    att = selection.split(val_separator)[0]
                    if att == source_attribute:
                        source_val = selection.split(val_separator)[1]
                    if att == target_attribute:
                        target_val = selection.split(val_separator)[1]
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
    sdf,
    filters,
    source_attribute,
    target_attribute,
    highlight_attribute,
    att_separator=";",
    val_separator=":",
) -> pd.DataFrame:
    edges = []
    att_groups = {}
    for f in filters:
        att, val = f.split(val_separator)
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
                    hatt, hval = highlight_attribute.split(val_separator)
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

def compute_top_attributes_query(
    query, sdf, adf, show_attributes, num_values, att_separator=";", val_separator=":"
) -> pd.DataFrame | Any:
    data_schema = get_data_schema(sdf)
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
    query, sdf, adf, time_attribute, time_series, att_separator=";", val_separator=":"
) -> pd.DataFrame:
    tdfs = []
    times = [t for t in sorted(sdf[time_attribute].unique()) if len(str(t)) > 0]
    for time in times:
        time_query = query + [{"attribute": time_attribute, "value": time}]
        tdf = compute_top_attributes_query(
            query=time_query,
            sdf=sdf,
            adf=adf,
            show_attributes=time_series,
            num_values=0,
            att_separator=att_separator,
            val_separator=val_separator,
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
