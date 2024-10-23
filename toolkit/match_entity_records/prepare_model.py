# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import polars as pl

from toolkit.match_entity_records.classes import AttributeToMatch, RecordsModel


def format_model_df(
    model: RecordsModel,
    max_rows: int = 0,
) -> pl.DataFrame:
    """
    Format the dataset for training the model
    :param model: The model to format
    :param max_rows: The maximum number of rows to return
    :return: The formatted dataset
    """

    if model.dataframe.is_empty():
        return pl.DataFrame()

    if not model.id_column:
        selected_df = model.dataframe.with_row_index(name="Entity ID")
    else:
        selected_df = model.dataframe.rename({model.id_column: "Entity ID"})

    selected_df = selected_df.rename({model.name_column: "Entity name"})
    selected_df = selected_df.with_columns([pl.col("Entity ID").cast(pl.Utf8)])

    selected_df = selected_df.select(
        ["Entity ID", "Entity name", *sorted(model.columns)]
    )
    if max_rows > 0:
        selected_df = selected_df.head(max_rows)
    return selected_df


def build_attribute_options(matching_dfs: dict[str, pl.DataFrame]) -> list[str]:
    attr_options = []
    skip_columns = ["Entity ID", "Entity name"]
    for dataset, merged_df in matching_dfs.items():
        attr_options.extend(
            [f"{c}::{dataset}" for c in merged_df.columns if c not in skip_columns]
        )
    return sorted(attr_options)


def build_attributes_list(attr_list: list[AttributeToMatch]) -> dict:
    df_renamed = defaultdict(dict)
    for attr in attr_list:
        att_name = attr.get("label")
        columns = attr.get("columns")
        if not columns:
            continue
        if not att_name:
            att_name = sorted(columns)[0].split("::")[0]
        for val in columns:
            col, dataset = val.split("::")
            if dataset not in df_renamed:
                df_renamed[dataset] = {}
            df_renamed[dataset][col] = att_name

    return df_renamed