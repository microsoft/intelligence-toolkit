# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


from itertools import product

import pandas as pd
import polars as pl


def create_window_df(
    groups: list[str], temporal: str, aggregates: list[str], wdf: pl.DataFrame
) -> pl.DataFrame:
    ldf = wdf.melt(
        id_vars=[*groups, temporal],
        value_vars=aggregates,
        variable_name="Attribute",
        value_name="Value",
    )

    # transform attribute and value columns into string
    ldf = ldf.with_columns(
        pl.col("Attribute").cast(pl.String), pl.col("Value").cast(pl.String)
    )
    ldf = ldf.with_columns(
        (pl.col("Attribute") + ":" + pl.col("Value")).alias("Attribute Value")
    )

    # Group by groups and count attribute values
    return (
        ldf.group_by([*groups, temporal, "Attribute Value"])
        .agg(pl.len().alias(f"{temporal} Window Count"))
        .sort([*groups, temporal, "Attribute Value"])
    )


def process_group(ldf, temporal, group) -> pl.DataFrame:
    group_dict = group.to_dict(as_series=False)
    new_rows = []
    unique_att_vals = ldf["Attribute Value"].unique()
    unique_time_vals = ldf["temporal"].unique()
    product_combinations = list(product(unique_time_vals, unique_att_vals))

    for time_val, att_val in product_combinations:
        if not (
            (group_dict["temporal"] == time_val)
            & (group_dict[f"{temporal} Window Count"] == att_val)
        ).any():
            new_rows.append(
                {
                    "groups": group_dict["groups"][
                        0
                    ],  # Assuming `groups` is a single value per group
                    "temporal": time_val,
                    "Attribute Value": att_val,
                    f"{temporal} Window Count": 0,
                }
            )
    return pl.DataFrame(new_rows)


def rank_temporal_attributes(
    ldf: pl.DataFrame, groups: list[str], temporal_atts: list[str], temporal: str
) -> pl.DataFrame:
    tdfs = []

    # Process each group and collect results
    results = ldf.groupby("groups").apply(lambda x: process_group(ldf, temporal, x))

    # Concatenate the original DataFrame with the new rows
    ldf = pl.concat([ldf, results])

    # Optionally, sort `ldf` if needed
    ldf = ldf.sort(["groups", "temporal", "Attribute Value"])

    # Calculate deltas in counts within each group and attribute value
    for _, ddf in ldf.groupby([*groups, "Attribute Value"]):
        ldf.loc[ddf.index, f"{temporal} Window Delta"] = (
            ddf[f"{temporal} Window Count"].diff().fillna(0)
        )
    for tatt in temporal_atts:
        tdf = ldf[ldf[temporal] == tatt].copy(deep=True)
        # rank counts for each attribute value
        for att_val in tdf["Attribute Value"].unique():
            tdf.loc[
                (tdf["Attribute Value"] == att_val),
                f"{temporal} Window Rank",
            ] = tdf[tdf["Attribute Value"] == att_val][f"{temporal} Window Count"].rank(
                ascending=False, method="first"
            )
        tdfs.append(tdf)
    return pl.concat(tdfs).sort(by=temporal)


def temp_rank(
    ldf: pd.DataFrame, groups: list[str], temporal_atts: list[str], temporal: str
) -> pd.DataFrame:
    tdfs = []
    for name, group in ldf.groupby(groups):
        for att_val in ldf["Attribute Value"].unique():
            for time_val in ldf[temporal].unique():
                if (
                    len(
                        group[
                            (group[temporal] == time_val)
                            & (group[f"{temporal} Window Count"] == att_val)
                        ]
                    )
                    == 0
                ):
                    ldf.loc[len(ldf)] = [
                        *name,
                        time_val,
                        att_val,
                        0,
                    ]

    # Calculate deltas in counts within each group and attribute value
    for _, ddf in ldf.groupby([*groups, "Attribute Value"]):
        ldf.loc[ddf.index, f"{temporal} Window Delta"] = (
            ddf[f"{temporal} Window Count"].diff().fillna(0)
        )
    for tatt in temporal_atts:
        tdf = ldf[ldf[temporal] == tatt].copy(deep=True)
        # rank counts for each attribute value
        for att_val in tdf["Attribute Value"].unique():
            tdf.loc[
                (tdf["Attribute Value"] == att_val),
                f"{temporal} Window Rank",
            ] = tdf[tdf["Attribute Value"] == att_val][f"{temporal} Window Count"].rank(
                ascending=False, method="first"
            )
        tdfs.append(tdf)
    return pd.concat(tdfs).sort_values(by=temporal)
