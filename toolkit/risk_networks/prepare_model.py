# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import re
from collections import defaultdict
from typing import Any

import networkx as nx
import polars as pl

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.risk_networks.config import ENTITY_LABEL, FlagAggregatorType


def clean_text(text: str | int) -> str:
    # remove punctuation but retain characters and digits in any language
    # compress whitespace to single space
    cleaned_text = re.sub(r"[^\w\s&@\+]", "", str(text)).strip()
    # cleaned_text = re.sub(r"[^\w\s&@+/]", "", str(text)).strip()
    return re.sub(r"\s+", " ", cleaned_text)


def format_data_columns(
    values_df: pl.DataFrame, columns_to_link: list[str], entity_id_column: str | int
) -> pl.DataFrame:
    values_df = values_df.with_columns(
        [
            pl.col(entity_id_column)
            .map_elements(clean_text, return_dtype=pl.Utf8)
            .alias(entity_id_column)
        ]
    )
    for value_col in columns_to_link:
        values_df = values_df.with_columns(
            [
                pl.col(value_col)
                .map_elements(clean_text, return_dtype=pl.Utf8)
                .alias(value_col)
            ]
        )
    return values_df


def generate_attribute_links(
    data_df: pl.DataFrame,
    entity_id_column: str,
    columns_to_link: list[str],
    existing_links: list | None = None,
) -> list:
    """
    Generate attribute links for the given entity and columns.

    Args:
        data_df (pl.DataFrame): The DataFrame containing the data.
        entity_id_column (str): The name of the column containing entity IDs.
        columns_to_link (list[str]): A list of column names to link as attributes.
        existing_links (list, optional): Existing attribute links. Defaults to None.

    Returns:
        list: A list of attribute links.
    """
    attribute_links = existing_links or []

    for value_col in columns_to_link:
        data_df = data_df.with_columns([pl.lit(value_col).alias("attribute_col")])

        attribute_links.append(
            data_df.select([entity_id_column, "attribute_col", value_col])
            .to_numpy()
            .tolist()
        )

    return attribute_links


def build_main_graph(
    attribute_links: list[Any] | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    if attribute_links is None:
        return graph

    value_to_atts = defaultdict(set)
    for link_list in attribute_links:
        for link in link_list:
            n1 = f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}{link[0]}"
            n2 = f"{link[1]}{ATTRIBUTE_VALUE_SEPARATOR}{link[2]}"
            edge = (n1, n2) if n1 < n2 else (n2, n1)
            graph.add_edge(edge[0], edge[1], type=link[1])
            graph.add_node(n1, type=ENTITY_LABEL)
            graph.add_node(n2, type=link[1])
            value_to_atts[link[2]].add(n2)

    for atts in value_to_atts.values():
        att_list = list(atts)
        for i, att1 in enumerate(att_list):
            for att2 in att_list[i + 1 :]:
                edge = (att1, att2) if att1 < att2 else (att2, att1)
                graph.add_edge(edge[0], edge[1], type="equality")
    return graph


def build_flag_links(
    df_flag: pl.DataFrame,
    entity_col: str,
    flag_agg: FlagAggregatorType,
    flag_columns: list[str],
    existing_flag_links: list | None = None,
) -> list[Any]:
    flag_links = existing_flag_links or []

    if entity_col not in df_flag.columns:
        msg = f"Column {entity_col} not found in the DataFrame."
        raise ValueError(msg)

    for value_col in flag_columns:
        if value_col not in df_flag.columns:
            msg = f"Column {value_col} not found in the DataFrame."
            raise ValueError(msg)
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
            flag_links.extend([[val[0], value_col, val[1], 1] for val in vals])
        elif flag_agg == FlagAggregatorType.Count.value:
            flag_links.extend([[val[0], value_col, value_col, val[1]] for val in vals])

    return flag_links


def transform_entity(entity):
    return f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}{entity}"


def build_flags(
    network_flag_links: list | None = None,
) -> tuple:
    if network_flag_links is None:
        return pl.DataFrame(), 0, 0

    flags = pl.DataFrame(
        {
            "entity": [item[0] for item in network_flag_links],
            "type": [item[1] for item in network_flag_links],
            "flag": [item[2] for item in network_flag_links],
            "count": [item[3] for item in network_flag_links],
        }
    )
    flags = flags.group_by(["entity", "type", "flag"]).agg(pl.sum("count"))
    flags = flags.with_columns(
        [flags["entity"].map_elements(transform_entity).alias("qualified_entity")]
    )
    overall_df = flags.group_by("qualified_entity").agg(pl.sum("count"))
    max_entity_flags = overall_df["count"].max()
    mean_flagged_flags = round(
        overall_df.filter(pl.col("count") > 0)["count"].mean(), 2
    )

    return flags, max_entity_flags, mean_flagged_flags


def build_groups(
    value_cols: list[str],
    df_groups: pl.DataFrame,
    entity_col: str,
    existing_group_links: list | None = None,
) -> list[Any]:
    group_links = existing_group_links or []

    if df_groups.is_empty():
        return group_links

    for value_col in value_cols:
        if value_col not in df_groups.columns:
            msg = f"Column {value_col} not found in the DataFrame."
            raise ValueError(msg)

        df_groups = df_groups.with_columns([pl.lit(value_col).alias("attribute_col")])
        if entity_col not in df_groups.columns:
            msg = f"Column {entity_col} not found in the DataFrame."
            raise ValueError(msg)

        link_list = (
            df_groups.select([entity_col, "attribute_col", value_col])
            .to_numpy()
            .tolist()
        )
        group_links.append(link_list)

    return group_links


def build_model_with_attributes(
    input_dataframe: pl.DataFrame, entity_id_column: str, columns_to_link: list[str]
) -> nx.Graph:
    data_df = format_data_columns(input_dataframe, columns_to_link, entity_id_column)
    attribute_links = generate_attribute_links(
        data_df, entity_id_column, columns_to_link
    )

    return build_main_graph(attribute_links)


def get_flags(
    flags_dataframe, entity_col, flag_agg, value_cols
) -> tuple[pl.DataFrame, int, int]:
    flag_links = build_flag_links(
        flags_dataframe,
        entity_col,
        flag_agg,
        value_cols,
    )
    return build_flags(flag_links)
