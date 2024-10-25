# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import polars as pl

import toolkit.AI.utils as utils
from toolkit.AI.client import OpenAIClient
from toolkit.compare_case_groups import prompts
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
from toolkit.helpers.classes import IntelligenceWorkflow


class CompareCaseGroups(IntelligenceWorkflow):
    model_df = pl.DataFrame()
    filtered_df = pl.DataFrame()
    prepared_df = pl.DataFrame()

    def __init__(self):
        self.filters = []
        self.groups = []
        self.aggregates = []
        self.temporal = ""

    def get_dataset_proportion(self) -> int:
        initial_row_count = len(self.prepared_df)
        filtered_row_count = len(self.filtered_df)
        return round(
            100 * filtered_row_count / initial_row_count
            if initial_row_count > 0
            else 0,
            0,
        )

    def get_filter_options(self, input_df: pl.DataFrame) -> list[str]:
        sorted_atts = []
        sorted_cols = sorted(input_df.columns)
        for col in sorted_cols:
            unique_sorted_values = (
                input_df.with_columns(pl.col(col).cast(pl.Utf8))  # Cast to string
                .select(pl.col(col).unique())  # Get unique values
                .to_series()  # Convert to Series
                .sort()  # Sort the unique values
            )
            vals = [
                f"{col}:{x}"
                for x in unique_sorted_values
                if x
                not in [
                    "",
                    "<NA>",
                    "nan",
                    "NaN",
                    "None",
                    "none",
                    "NULL",
                    "null",
                ]
            ]
            sorted_atts.extend(vals)
        return sorted_atts

    def _select_columns_ranked_df(self, ranked_df: pl.DataFrame) -> None:
        columns = [g.lower() for g in self.groups]
        default_columns = [
            "group_count",
            "group_rank",
            "attribute_value",
            "attribute_count",
            "attribute_rank",
        ]

        columns.extend(default_columns)

        if self.temporal:
            columns.extend(
                [
                    f"{self.temporal}_window",
                    f"{self.temporal}_window_count",
                    f"{self.temporal}_window_rank",
                    f"{self.temporal}_window_delta",
                ]
            )

        self.model_df = ranked_df.select(columns)

    def create_data_summary(
        self,
        prepared_df: pl.DataFrame,
        filters: list[str],
        groups: list[str],
        aggregates: list[str],
        temporal: str = "",
    ):
        self.filters = filters
        self.groups = groups
        self.aggregates = aggregates
        self.temporal = temporal
        self.prepared_df = prepared_df

        self.prepared_df = self.prepared_df.drop_nulls(subset=self.groups)

        self.model_df = self.prepared_df.with_columns(
            [
                pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
                for col in self.prepared_df.columns
            ]
        )

        self.filtered_df = filter_df(self.model_df, filters)

        grouped_df = build_grouped_df(self.filtered_df, groups)

        attributes_df = build_attribute_df(self.filtered_df, groups, aggregates)

        temporal_df = pl.DataFrame()
        if temporal:
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

    def get_report_data(
        self,
        selected_groups=None,
        top_group_ranks=None,
    ) -> tuple[pl.DataFrame, str]:
        selected_df = self.model_df

        filter_description = ""
        if selected_groups:
            selected_df = selected_df.filter(pl.col(self.groups).is_in(selected_groups))
            filter_description = f'Filtered to the following groups only: {", ".join([str(s) for s in selected_groups])}'
        elif top_group_ranks:
            selected_df = selected_df.filter(pl.col("group_rank") <= top_group_ranks)
            filter_description = (
                f"Filtered to the top {top_group_ranks} groups by record count"
            )
        return selected_df, filter_description

    def generate_group_report(
        self,
        report_data: pl.DataFrame,
        filter_description=str,
        ai_instructions=prompts.user_prompt,
        callbacks=[],
    ):
        variables = {
            "description": self.get_summary_description(),
            "dataset": report_data.write_csv(),
            "filters": filter_description,
        }

        messages = utils.generate_messages(
            ai_instructions,
            prompts.list_prompts["report_prompt"],
            variables,
            prompts.list_prompts["safety_prompt"],
        )
        return OpenAIClient(self.ai_configuration).generate_chat(
            messages, callbacks=callbacks
        )
