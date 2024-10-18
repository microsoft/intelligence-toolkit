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
    model_df = pl.DataFrame()
    filtered_df = pl.DataFrame()
    final_df = pl.DataFrame()

    def __init__(self):
        self.filters = []
        self.groups = []
        self.aggregates = []
        self.temporal = ""

    def get_dataset_proportion(self) -> int:
        initial_row_count = len(self.model_df)
        filtered_row_count = len(self.filtered_df)
        return round(
            100 * filtered_row_count / initial_row_count
            if initial_row_count > 0
            else 0,
            0,
        )

    def _select_columns_ranked_df(self, ranked_df: pl.DataFrame) -> None:
        lower_groups = [g.lower() for g in self.groups]

        if self.temporal != "":
            columns = [
                *lower_groups,
                "group_count",
                "group_rank",
                "attribute_value",
                "attribute_count",
                "attribute_rank",
                f"{self.temporal}_window",
                f"{self.temporal}_window_count",
                f"{self.temporal}_window_rank",
                f"{self.temporal}_window_delta",
            ]
        else:
            columns = [
                *lower_groups,
                "group_count",
                "group_rank",
                "attribute_value",
                "attribute_count",
                "attribute_rank",
            ]

        self.model_df = ranked_df.select(columns)

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

        self.final_df = self.final_df.drop_nulls(subset=self.groups)

        self.model_df = self.final_df.with_columns(
            [
                pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
                for col in self.final_df.columns
            ]
        )

        self.filtered_df = filter_df(self.model_df, filters)

        grouped_df = build_grouped_df(self.filtered_df, groups)

        attributes_df = build_attribute_df(self.filtered_df, groups, aggregates)

        temporal_df = pl.DataFrame()
        if temporal is not None and temporal != "":
            window_df = create_window_df(groups, temporal, aggregates, self.filtered_df)

            temporal_atts = sorted(self.model_df[temporal].cast(pl.Utf8).unique())

            temporal_df = build_temporal_data(
                window_df, groups, temporal_atts, temporal
            )

        ranked_df = build_ranked_df(
            temporal_df,
            grouped_df,
            attributes_df,
            temporal,
            groups,
        )
        self._select_columns_ranked_df(ranked_df)

    def _format_list(self, items, bold=True, escape_colon=False) -> str:
        formatted_items = []
        for item in items:
            if escape_colon:
                item = item.replace(":", "\\:")
            if bold:
                item = f"**{item}**"
            formatted_items.append(item)
        return "[" + ", ".join(formatted_items) + "]"

    def get_summary_description(self) -> str:
        groups_text = self._format_list(self.groups)
        filters_text = self._format_list(self.filters, escape_colon=True)

        description_lines = ["This table shows:"]

        if self.filters:
            description_lines.append(
                f"- A summary of **{len(self.filtered_df)}** data records matching {filters_text}, representing **{self.get_dataset_proportion()}%** of the overall dataset with values for all grouping attributes"
            )
        else:
            description_lines.append(
                f"- A summary of all **{len(self.filtered_df)}** data records with values for all grouping attributes"
            )

        description_lines.extend(
            [
                f"- The **group_count** of records for all {groups_text} groups, and corresponding **group_rank**",
                f"- The **attribute_count** of each **attribute_value** for all {groups_text} groups, and corresponding **attribute_rank**",
            ]
        )

        if self.temporal:
            description_lines.extend(
                [
                    f"- The **{self.temporal}_window_count** of each **attribute_value** for each **{self.temporal}_window** for all {groups_text} groups, and corresponding **{self.temporal}_window_rank**",
                    f"- The **{self.temporal}_window_delta**, or change in the **attribute_value_count** for successive **{self.temporal}_window** values, within each {groups_text} group",
                ]
            )

        return "\n".join(description_lines)
