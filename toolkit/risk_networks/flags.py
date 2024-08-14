# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import json
from collections import defaultdict

import networkx as nx
import polars as pl

import toolkit.risk_networks.config as config
from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.risk_networks.config import FlagAggregatorType


def integrate_flags(graph: nx.Graph, df_integrated_flags: pl.DataFrame) -> nx.Graph:
    if not graph.nodes() or df_integrated_flags.is_empty():
        return nx.Graph()

    df_integrated_flags = df_integrated_flags.filter(pl.col("count") > 0)
    flagged_nodes = (
        df_integrated_flags.select("qualified_entity").unique().to_series().to_list()
    )

    flagged_nodes = [node for node in flagged_nodes if node in graph.nodes()]
    for node in flagged_nodes:
        graph.nodes[node]["flags"] = df_integrated_flags.filter(
            pl.col("qualified_entity") == node
        )["count"].sum()
    return graph


def transform_entity(entity):
    return f"{config.entity_label}{ATTRIBUTE_VALUE_SEPARATOR}{entity}"


def build_flags(
    network_flag_links: list,
) -> tuple:
    print("network_flag_links", network_flag_links)
    flags = pl.DataFrame(
        {
            "entity": [item[0] for item in network_flag_links],
            "type": [item[1] for item in network_flag_links],
            "flag": [item[2] for item in network_flag_links],
            "count": [item[3] for item in network_flag_links],
        }
    )
    flags = flags.groupby(["entity", "type", "flag"]).agg(pl.sum("count"))
    flags = flags.with_columns(
        [flags["entity"].apply(transform_entity).alias("qualified_entity")]
    )
    overall_df = flags.groupby("qualified_entity").agg(pl.sum("count"))
    max_entity_flags = overall_df["count"].max()
    mean_flagged_flags = round(
        overall_df.filter(pl.col("count") > 0)["count"].mean(), 2
    )

    return flags, max_entity_flags, mean_flagged_flags


def prepare_links(
    df_flag: pl.DataFrame,
    entity_col: str,
    flag_agg: FlagAggregatorType,
    flag_columns: list[str],
) -> list:
    model_links = []

    for value_col in flag_columns:
        gdf = df_flag.with_columns([pl.col(value_col).cast(pl.Int32).alias(value_col)])
        gdf = gdf.group_by(entity_col).agg([pl.sum(col) for col in flag_columns])
        vals = (
            gdf[
                [
                    entity_col,
                    value_col,
                ]
            ]
            .to_numpy()
            .tolist()
        )
        if flag_agg == FlagAggregatorType.Instance.value:
            gdf = gdf.with_columns([pl.lit(1).alias("count_col")])
            model_links.extend([[val[0], value_col, val[1], 1] for val in vals])
        elif flag_agg == FlagAggregatorType.Count.value:
            model_links.extend([[val[0], value_col, value_col, val[1]] for val in vals])

    return [model_links]


def build_exposure_data(
    integrated_flags: pl.DataFrame,
    c_nodes: list[str],
    selected_entity: str,
    graph: nx.Graph,
):
    if integrated_flags.is_empty():
        return ""

    qualified_selected = (
        f"{config.entity_label}{ATTRIBUTE_VALUE_SEPARATOR}{selected_entity}"
    )
    rdf = integrated_flags
    rdf = rdf.filter(pl.col("qualified_entity").is_in(c_nodes))
    rdf = rdf.groupby(["qualified_entity", "flag"]).agg(pl.col("count").sum())
    all_flagged = (
        rdf.filter(pl.col("count") > 0)
        .select("qualified_entity")
        .unique()
        .to_series()
        .to_list()
    )

    target_flags = (
        rdf.filter(pl.col("qualified_entity") == qualified_selected)
        .select(pl.col("count").sum())
        .item()
    )
    total_flags = rdf.select(pl.col("count").sum()).item()
    net_flags = total_flags - target_flags
    net_flagged = len(all_flagged)
    if qualified_selected in all_flagged:
        net_flagged -= 1

    steps_list = []
    nodes = []
    for flagged in all_flagged:
        all_paths = [
            list(x) for x in nx.all_shortest_paths(graph, flagged, qualified_selected)
        ]
        for path in all_paths:
            path_steps_list = []
            if len(path) <= 1:
                continue

            for _, step in enumerate(path):
                if config.entity_label in step:
                    step_risks = rdf.filter(pl.col("qualified_entity") == step)[
                        "count"
                    ].sum()

                    if step_risks == 0:
                        continue
                    node_flag = {"node": step, "flags": step_risks}
                else:
                    step_entities = nx.degree(graph, step)
                    if step_risks == 0:
                        continue
                    node_flag = {"node": step, "entities": step_entities}

                if node_flag not in nodes:
                    nodes.append(node_flag)

            for j, step in enumerate(path):
                if j < len(path) - 1:
                    source = step
                    destination = path[j + 1]
                    step1 = {"source": source, "target": destination}
                    path_steps_list.append(step1)
            steps_list.append(path_steps_list)

    path_items = defaultdict(list)
    paths = []
    for step in steps_list:
        source = step[0]["source"]
        path = step[1:]
        if len(path) == 0:
            path = [{"target": step[0]["target"]}]
        path_items[json.dumps(path)].append(source)

    for path, sources in path_items.items():
        path_list = []
        path_list.append(sources)

        for ixx, node in enumerate(json.loads(path)):
            if ixx == 0 and "source" in node:
                path_list.append([node["source"]])
            path_list.append([node["target"]])

        paths.append(path_list)

    flags_summary_count = {
        "direct": target_flags,
        "indirect": net_flags,
        "paths": len(paths),
        "entities": net_flagged,
    }
    return flags_summary_count, paths, nodes


def build_exposure_report(
    integrated_flags: pl.DataFrame,
    selected_entity: str,
    c_nodes: list[str],
    graph: nx.Graph,
) -> str:
    selected_data, all_paths, nodes = build_exposure_data(
        integrated_flags,
        c_nodes,
        selected_entity,
        graph,
    )
    context = "##### Risk Exposure Report\n\n"
    context += f"The selected entity **{selected_entity}** has **{selected_data['direct']}** direct flags and is linked to **{selected_data['indirect']}** indirect flags via **{selected_data['paths']}** paths from **{selected_data['entities']}** related entities:\n\n"

    for i, path in enumerate(all_paths):
        context += f"**Path {i + 1}**\n\n```\n"
        for ix, node in enumerate(path):
            indent = "".join(["  "] * ix)
            for step in node:
                node_value = [val for val in nodes if val["node"] == step]
                if config.entity_label in step:
                    step = f"{step} [linked to {node_value[0]['flags']} flags]"
                else:
                    step = f"{step} [linked to {node_value[0]['entities']} entities]"
                context += f"{indent}{step}\n"
            if ix < len(path) - 1:
                context += f"{indent}--->\n"

    return context + "\n```\n\n"
