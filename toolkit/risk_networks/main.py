# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl

from toolkit.risk_networks.prepare_model import (
    build_flag_links,
    build_flags,
    build_main_graph,
    format_data_columns,
    generate_attribute_links,
)


def build_model_with_attributes(
    input_dataframe: pl.DataFrame, entity_id_column: str, columns_to_link: list[str]
) -> nx.Graph:
    data_df = format_data_columns(input_dataframe, columns_to_link, entity_id_column)
    attribute_links = generate_attribute_links(
        data_df, entity_id_column, columns_to_link
    )

    return build_main_graph(network_attribute_links=attribute_links)


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
