# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pandas as pd
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
        attribute_df, on=[*groups, "Attribute Value"], how="left", suffix="_r"
    )

    odf = odf.sort(by=groups, descending=False)

    if temporal != "":
        odf = odf.with_columns(
            [
                pl.col(temporal).alias(f"{temporal} Window"),
                pl.col(f"{temporal} Window Rank").cast(pl.Int32),
                pl.col(f"{temporal} Window Delta").cast(pl.Int32),
            ]
        )

    return odf.with_columns(
        [pl.col("Attribute Rank").cast(pl.Int32), pl.col("Group Rank").cast(pl.Int32)]
    )


def build_grouped_df(main_dataset, groups) -> pd.DataFrame:
    gdf = main_dataset.melt(
        id_vars=groups,
        value_vars=["Subject ID"],
        var_name="Attribute",
        value_name="Value",
    )

    gdf["Attribute Value"] = gdf["Attribute"] + ":" + gdf["Value"]
    gdf = gdf.groupby(groups).size().reset_index(name="Group Count")
    # Add group ranks
    gdf["Group Rank"] = gdf["Group Count"].rank(
        ascending=False, method="max", na_option="bottom"
    )
    return gdf


def build_attribute_df_pd(filtered_df, groups, aggregates=""):
    ndf = filtered_df.melt(
        id_vars=groups,
        value_vars=aggregates,
        var_name="Attribute",
        value_name="Value",
    )
    ndf.dropna(subset=["Value"], inplace=True)
    ndf["Attribute Value"] = ndf.apply(
        lambda x: str(x["Attribute"]) + ":" + str(x["Value"]), axis=1
    )

    attributes_df = (
        ndf.groupby([*groups, "Attribute Value"])
        .size()
        .reset_index(name="Attribute Count")
    )
    # Ensure all groups have entries for all attribute value
    grouped = attributes_df.groupby(groups)
    for name, group in grouped:
        for att_val in attributes_df["Attribute Value"].unique():
            # count rows with this group and attribute value
            row_count = len(group[group["Attribute Value"] == att_val])
            if row_count == 0:
                attributes_df.loc[len(attributes_df)] = [*name, att_val, 0]

    for att_val in attributes_df["Attribute Value"].unique():
        attributes_df.loc[
            attributes_df["Attribute Value"] == att_val, "Attribute Rank"
        ] = attributes_df[attributes_df["Attribute Value"] == att_val][
            "Attribute Count"
        ].rank(ascending=False, method="max", na_option="bottom")
    return attributes_df


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

    # Create "Attribute Value" column
    ndf = ndf.with_columns(
        (pl.col("Attribute") + ":" + pl.col("Value").cast(str)).alias("Attribute Value")
    )

    # Group by and count the occurrences
    attributes_df = ndf.group_by([*groups, "Attribute Value"]).agg(
        pl.len().alias("Attribute Count")
    )

    # Ensure all groups have entries for all attribute values
    all_attribute_values = attributes_df["Attribute Value"].unique().to_list()
    groups_df = filtered_df.select(groups).unique()
    all_combinations = pl.DataFrame({col: groups_df[col] for col in groups}).join(
        pl.DataFrame({"Attribute Value": all_attribute_values}), how="cross"
    )
    attributes_df = all_combinations.join(
        attributes_df, on=[*groups, "Attribute Value"], how="left"
    ).fill_null(0)

    # Calculate the rank
    return attributes_df.with_columns(
        [
            pl.col("Attribute Count")
            .rank("max", descending=True)
            .over("Attribute Value")
            .alias("Attribute Rank")
        ]
    )


def filter_df(main_df, filters: list[str]):
    filtered_df = main_df.copy()

    if len(filters) == 0:
        return filtered_df
        # filter to only those rows that match the filters
    for f in filters:
        col, val = f.split(":")
        filtered_df = filtered_df[filtered_df[col] == val]

    return filtered_df
