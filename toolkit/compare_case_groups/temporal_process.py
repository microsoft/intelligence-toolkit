# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


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

    ldf = ldf.filter(pl.col("Value").is_not_null())
    for group in groups:
        ldf = ldf.filter(pl.col(group).is_not_null())
    print(ldf)

    # Group by groups and count attribute values
    return (
        ldf.group_by([*groups, temporal, "Attribute Value"])
        .agg(pl.len().alias(f"{temporal} Window Count"))
        .sort([*groups, temporal, "Attribute Value"])
    )

# Calculate deltas in counts within each group and attribute value
def calculate_window_delta(temporal_df: pl.DataFrame, temporal) -> pl.DataFrame:
    return temporal_df.with_columns(
        (pl.col(f"{temporal} Window Count").diff().fill_null(0)).alias(
            f"{temporal} Window Delta"
        )
    )

def build_temporal_count(
    ldf: pl.DataFrame, groups: list[str], temporal: str
) -> pl.DataFrame:
    grouped_df = ldf.group_by(groups)
    for name, group in grouped_df:
        for att_val in ldf["Attribute Value"].unique():
            for time_val in ldf[temporal].unique():
                filtered_df = group.filter(
                    (pl.col(temporal) == time_val)
                    & (
                        pl.col(f"{temporal} Window Count").cast(pl.String)
                        == str(att_val)
                    )
                )
                if filtered_df.height == 0:
                    # Create a new row as a DataFrame
                    new_row = pl.DataFrame(
                        [[*name, time_val, att_val, 0]],  # type: ignore
                        schema=ldf.schema,
                    )

                    # Append the new row to ldf
                    ldf = ldf.vstack(new_row)

    return calculate_window_delta(ldf, temporal)


def build_temporal_data(
    ldf: pl.DataFrame, groups: list[str], temporal_atts: list[str], temporal: str
) -> pl.DataFrame:
    tdfs = []
    if ldf.shape[0] == 0:
        return ldf
    ldf = build_temporal_count(ldf, groups, temporal)
    if ldf.is_empty():
        return ldf

    for tatt in temporal_atts:
        tdf = ldf.filter(pl.col(temporal) == tatt)
        tdf = tdf.with_columns(pl.lit(0).alias(f"{temporal} Window Rank"))
        for att_val in tdf.select("Attribute Value").unique().to_numpy():
            filtered_df = tdf.filter(pl.col("Attribute Value") == att_val)
            ranked_series = filtered_df[f"{temporal} Window Count"].rank(
                descending=True, method="dense"
            )
            filtered_ranked = filtered_df.with_columns(
                ranked_series.alias(f"{temporal} Window Rank")
            )
            tdf = tdf.join(
                filtered_ranked,
                on=[*groups, temporal, "Attribute Value"],
                how="left",
                suffix=("_r"),
            )
            tdf = tdf.with_columns(
                pl.when(pl.col("Attribute Value") == att_val)
                .then(pl.col(f"{temporal} Window Rank_r"))
                .otherwise(pl.col(f"{temporal} Window Rank"))
                .alias(f"{temporal} Window Rank")
            )

            tdf = tdf.drop(
                [
                    f"{temporal} Window Delta_r",
                    f"{temporal} Window Rank_r",
                    f"{temporal} Window Count_r",
                ]
            )
        tdf = tdf.with_columns(
            pl.col(f"{temporal} Window Rank").cast(pl.Int64),
            pl.col(f"{temporal} Window Count").cast(pl.Int64),
        )
        tdfs.append(tdf)
    return pl.concat(tdfs).sort(by=temporal)
