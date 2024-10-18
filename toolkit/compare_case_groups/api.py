# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import pandas as pd
import polars as pl

from toolkit.compare_case_groups.build_dataframes import (
    build_attribute_df,
    build_grouped_df,
    build_ranked_df,
    filter_df,
)
from toolkit.compare_case_groups.temporal_process import (
    build_temporal_data,
    create_window_df,
)


class CompareCaseGroups:
    model_df = pd.DataFrame()
    filtered_df = pd.DataFrame()
    final_df = pd.DataFrame()

    def __init__(self):
        self.filters = []
        self.groups = []
        self.aggregates = []
        self.temporal = ""

    def get_dataset_proportion(self) -> int:
        initial_row_count = len(self.filtered_df)
        return round(
            100 * initial_row_count / initial_row_count if initial_row_count > 0 else 0,
            0,
        )

    def create_data_summary(
        self,
        final_df: pd.DataFrame,
        filters: list[str],
        groups: list[str],
        aggregates: list[str],
        temporal: str = "",
    ):
        self.filters = filters
        self.groups = groups
        self.aggregates = aggregates
        self.temporal = temporal
        self.final_df = final_df

        self.final_df = self.final_df.dropna(subset=self.groups)

        self.model_df = self.final_df.copy(deep=True)

        self.model_df = self.model_df.replace("", None)

        print(type(self.model_df))
        filtered_df = filter_df(self.model_df, filters)

        grouped_df = build_grouped_df(filtered_df, groups)

        attributes_df = build_attribute_df(
            pl.from_pandas(filtered_df), groups, aggregates
        )

        temporal_df = pl.DataFrame()
        temporal_atts = []
        # create Window df
        if temporal is not None and temporal != "":
            window_df = create_window_df(
                groups, temporal, aggregates, pl.from_pandas(filtered_df)
            )

            temporal_atts = sorted(self.model_df[temporal].astype(str).unique())

            temporal_df = build_temporal_data(
                window_df, groups, temporal_atts, temporal
            )
            # Create overall df
            ranked_df = build_ranked_df(
                temporal_df,
                pl.from_pandas(grouped_df),
                attributes_df,
                temporal or "",
                groups,
            ).to_pandas()

            self.model_df = (
                ranked_df[
                    [
                        *[g.lower() for g in groups],
                        "group_count",
                        "group_rank",
                        "attribute_value",
                        "attribute_count",
                        "attribute_rank",
                        f"{temporal}_window",
                        f"{temporal}_window_count",
                        f"{temporal}_window_rank",
                        f"{temporal}_window_delta",
                    ]
                ]
                if temporal != ""
                else ranked_df[
                    [
                        *[g.lower() for g in groups],
                        "group_count",
                        "group_rank",
                        "attribute_value",
                        "attribute_count",
                        "attribute_rank",
                    ]
                ]
            )

    def get_summary_description(
        self,
    ) -> str:
        groups_text = "[" + ", ".join(["**" + g + "**" for g in self.groups]) + "]"
        filters_text = (
            "["
            + ", ".join(["**" + f.replace(":", "\\:") + "**" for f in self.filters])
            + "]"
        )

        description = "This table shows:"

        description += (
            f"\n- A summary of **{len(self.filtered_df)}** data records matching {filters_text}, representing **{self.get_dataset_proportion()}%** of the overall dataset with values for all grouping attributes"
            if len(self.filters) > 0
            else f"\n- A summary of all **{len(self.filtered_df)}** data records with values for all grouping attributes"
        )
        description += f"\n- The **group_count** of records for all {groups_text} groups, and corresponding **group_rank**"
        description += f"\n- The **attribute_count** of each **attribute_value** for all {groups_text} groups, and corresponding **attribute_rank**"
        if self.temporal != "":
            description += f"\n- The **{self.temporal}_window_count** of each **attribute_value** for each **{self.temporal}_window** for all {groups_text} groups, and corresponding **{self.temporal}_window_rank**"
            description += f"\n- The **{self.temporal}_window_delta**, or change in the **attribute_value_count** for successive **{self.temporal}_window** values, within each {groups_text} group"
        return description
