# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl


def format_dataset(
    record_df: pl.DataFrame,
    entity_attribute_columns: list[str],
    entity_name_column: str,
    entity_id_column: str = "",
    max_rows: int = 0,
) -> pl.DataFrame:
    if record_df.is_empty():
        return pl.DataFrame()

    if entity_id_column == "":
        selected_df = record_df.with_row_index(name="Entity ID")
    else:
        selected_df = record_df.rename({entity_id_column: "Entity ID"})

    selected_df = selected_df.rename({entity_name_column: "Entity name"})
    selected_df = selected_df.with_columns([pl.col("Entity ID").cast(pl.Utf8)])

    selected_df = selected_df.select(
        ["Entity ID", "Entity name", *sorted(entity_attribute_columns)]
    )
    if max_rows > 0:
        selected_df = selected_df.head(max_rows)
    return selected_df
