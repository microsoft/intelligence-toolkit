# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import polars as pl


def build_ranked_df(
    temporal_df: pl.DataFrame,
    group_df: pl.DataFrame,
    attribute_df: pl.DataFrame,
    temporal: str,
    groups: list[str],
) -> pl.DataFrame:
    if temporal != "":
        odf = temporal_df.join(group_df, on=groups, how="left", suffix="_r")
    else:
        odf = attribute_df.join(group_df, on=groups, how="left", suffix="_r")

    odf = odf.join(
        attribute_df, on=[*groups, "attribute_value"], how="left", suffix="_r"
    )

    odf = odf.sort(by=groups)

    if temporal != "":
        odf = odf.with_columns(
            [
                pl.col(temporal).alias(f"{temporal}_window"),
                pl.col(f"{temporal}_window_rank").cast(pl.Int32),
                pl.col(f"{temporal}_window_delta").cast(pl.Int32),
            ]
        )

    return odf.with_columns(
        [pl.col("attribute_rank").cast(pl.Int32), pl.col("group_rank").cast(pl.Int32)]
    )


def build_grouped_df(main_dataset: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    """
    This function takes a main dataset and a list of grouping columns, then processes
    and returns a DataFrame with the counts of each group and their ranks.

    Parameters:
    main_dataset (pl.DataFrame): The main dataset to process.
    groups (list of str): The list of column names to group by.

    Returns:
    pl.DataFrame: A DataFrame with group counts and ranks.
    """
    # Ensure groups is a list of strings
    if not all(isinstance(group, str) for group in groups):
        error_text = "All elements in groups must be strings"
        raise ValueError(error_text)

    main_dataset = main_dataset.with_columns(
        pl.arange(0, main_dataset.height).cast(pl.Utf8).alias("record_id")
    )

    gdf = main_dataset.melt(
        id_vars=groups,
        value_vars=["record_id"],
        variable_name="Attribute",
        value_name="Value",
    )

    gdf = gdf.with_columns(
        (pl.col("Attribute") + ":" + pl.col("Value")).alias("attribute_value")
    )

    gdf = gdf.group_by(groups).agg(pl.len().alias("group_count"))

    gdf = gdf.with_columns(
        pl.col("group_count").rank(method="max", descending=True).alias("group_rank")
    )

    return gdf.sort(by=groups)

def build_attribute_df(
    filtered_df: pl.DataFrame, groups: list[str], aggregates: str = ""
) -> pl.DataFrame:

    ndf = filtered_df.melt(
        id_vars=groups,
        value_vars=aggregates,
        variable_name="Attribute",
        value_name="Value",
    )
    # Drop rows with NaN values in the "Value" column
    ndf = ndf.drop_nulls(subset=["Value"])
    # Create "attribute_value" column
    ndf = ndf.with_columns(
        (pl.col("Attribute") + ":" + pl.col("Value").cast(str)).alias("attribute_value")
    )

    # Group by and count the occurrences
    attributes_df = ndf.group_by([*groups, "attribute_value"]).agg(
        pl.len().alias("attribute_count")
    )
    # Ensure all groups have entries for all attribute_values
    all_attribute_values = attributes_df["attribute_value"].unique().to_list()
    groups_df = filtered_df.select(groups).unique()
    all_combinations = pl.DataFrame({col: groups_df[col] for col in groups}).join(
        pl.DataFrame({"attribute_value": all_attribute_values}), how="cross"
    )
    attributes_df = all_combinations.join(
        attributes_df, on=[*groups, "attribute_value"], how="left"
    ).fill_null(0)
    # Calculate the rank
    return attributes_df.with_columns(
        [
            pl.col("attribute_count")
            .rank("max", descending=True)
            .over("attribute_value")
            .alias("attribute_rank")
        ]
    )


def filter_df(main_df: pl.DataFrame, filters: list[str]) -> pl.DataFrame:
    if len(filters) == 0:
        return main_df

    # Filter to only those rows that match the filters
    for f in filters:
        col, val = f.split(":")
        main_df = main_df.filter(pl.col(col) == val)

    return main_df
