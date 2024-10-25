# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
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
        (pl.col("Attribute") + ":" + pl.col("Value")).alias("attribute_value")
    )

    ldf = ldf.filter(pl.col("Value").is_not_null())
    for group in groups:
        ldf = ldf.filter(pl.col(group).is_not_null())

    # Group by groups and count attribute values
    return (
        ldf.group_by([*groups, temporal, "attribute_value"])
        .agg(pl.len().alias(f"{temporal}_window_count"))
        .sort([*groups, temporal, "attribute_value"])
    )

# Calculate deltas in counts within each group and attribute value
def calculate_window_delta(
    groups: list[str], temporal_df: pl.DataFrame, temporal
) -> pl.DataFrame:
    return temporal_df.sort([*groups, temporal, "attribute_value"]).with_columns(
        [
            pl.col(f"{temporal}_window_count")
            .diff()
            .over([*groups, "attribute_value"])
            .fill_null(0)
            .alias(f"{temporal}_window_delta")
        ]
    )


def build_temporal_count(
    ldf: pl.DataFrame, groups: list[str], temporal: str
) -> pl.DataFrame:
    grouped_df = ldf.group_by(groups)

    unique_time_vals = ldf[temporal].unique()
    unique_att_vals = ldf["attribute_value"].unique()

    new_rows = []

    # Iterate over unique temporal values
    for time_val in unique_time_vals:
        for name, group in grouped_df:
            for att_val in unique_att_vals:
                # Filter the group based on temporal and attribute value
                filtered_df = group.filter(
                    (pl.col(temporal) == time_val)
                    & (pl.col("attribute_value") == att_val)
                )
                # Check if the filtered dataframe is empty
                if filtered_df.height == 0:
                    # Append the new row to the list
                    new_rows.append([*name, time_val, att_val, 0])
    # Convert new rows to a DataFrame and append to the original DataFrame
    if new_rows:
        new_rows_df = pl.DataFrame(new_rows, schema=ldf.schema)
        ldf = ldf.vstack(new_rows_df)

    # Remove attribute_values for groups where attribute_count are 0 for the window
    aggregated_df = ldf.group_by([*groups, "attribute_value"]).agg(
        [pl.col(f"{temporal}_window_count").sum().alias("total_window_count")]
    )
    zero_count_groups = aggregated_df.filter(pl.col("total_window_count") == 0)

    filtered_result = ldf.join(
        zero_count_groups, on=[*groups, "attribute_value"], how="anti"
    )

    return calculate_window_delta(groups, filtered_result, temporal)


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
        tdf = tdf.with_columns(pl.lit(0).alias(f"{temporal}_window_rank"))
        for att_val in tdf.select("attribute_value").unique().to_numpy():
            filtered_df = tdf.filter(pl.col("attribute_value") == att_val)
            ranked_series = filtered_df[f"{temporal}_window_count"].rank(
                descending=True, method="dense"
            )
            filtered_ranked = filtered_df.with_columns(
                ranked_series.alias(f"{temporal}_window_rank")
            )
            tdf = tdf.join(
                filtered_ranked,
                on=[*groups, temporal, "attribute_value"],
                how="left",
                suffix=("_r"),
            )
            tdf = tdf.with_columns(
                pl.when(pl.col("attribute_value") == att_val)
                .then(pl.col(f"{temporal}_window_rank_r"))
                .otherwise(pl.col(f"{temporal}_window_rank"))
                .alias(f"{temporal}_window_rank")
            )

            tdf = tdf.drop(
                [
                    f"{temporal}_window_delta_r",
                    f"{temporal}_window_rank_r",
                    f"{temporal}_window_count_r",
                ]
            )
        tdf = tdf.with_columns(
            pl.col(f"{temporal}_window_rank").cast(pl.Int64),
        )
        tdfs.append(tdf)
    return pl.concat(tdfs).sort(by=temporal)
