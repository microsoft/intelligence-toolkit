# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import math
import pandas as pd
import plotly.graph_objects as go
from pacsynth import (
    AccuracyMode,
    Dataset,
    DpAggregateSeededParametersBuilder,
    DpAggregateSeededSynthesizer,
    FabricationMode,
)

import app.util.df_functions as df_functions
import intelligence_toolkit.anonymize_case_data.queries as queries
import intelligence_toolkit.anonymize_case_data.visuals as visuals
from intelligence_toolkit.anonymize_case_data.error_report import ErrorReport
from intelligence_toolkit.anonymize_case_data.synthesizability_statistics import (
    SynthesizabilityStatistics,
)
from intelligence_toolkit.helpers.classes import IntelligenceWorkflow


class AnonymizeCaseData(IntelligenceWorkflow):
    def __init__(self) -> None:
        self.protected_number_of_records = 0
        self.delta = 0
        self.sensitive_df = pd.DataFrame()
        self.aggregate_df = pd.DataFrame()
        self.synthetic_df = pd.DataFrame()
        self.aggregate_error_report = pd.DataFrame()
        self.synthetic_error_report = pd.DataFrame()

    def analyze_synthesizability(self, df: pd.DataFrame) -> SynthesizabilityStatistics:
        distinct_counts = []
        att_cols = list(df.columns)
        num_cols = len(att_cols)
        for col in df.columns.to_numpy():
            distinct_values = [
                x for x in df[col].astype(str).unique() if x not in ["", "nan"]
            ]
            num = len(distinct_values)
            if num > 0:
                distinct_counts.append(num)
        distinct_counts.sort()
        overall_att_count = sum(distinct_counts)
        possible_combinations = math.prod(distinct_counts)
        possible_combinations_per_row = round(possible_combinations / len(df), 1)
        mean_vals_per_record = (
            sum(
                [
                    len([y for y in x if str(y) not in ["nan", ""]])
                    for x in df.to_numpy()
                ]
            )
            / df.shape[0]
        )
        max_combinations_per_record = 2**mean_vals_per_record
        excess_combinations_ratio = (
            possible_combinations_per_row / max_combinations_per_record
        )
        return SynthesizabilityStatistics(
            num_cols,
            overall_att_count,
            possible_combinations,
            possible_combinations_per_row,
            mean_vals_per_record,
            max_combinations_per_record,
            excess_combinations_ratio,
        )

    def anonymize_case_data(
        self,
        df: pd.DataFrame,
        epsilon: float,
        reporting_length: int = 4,
        percentile_percentage: float = 99,
        percentile_epsilon_proportion: float = 0.01,
        number_of_records_epsilon_proportion: float = 0.005,
        weight_selection_percentile: float = 95,
        accuracy_mode: AccuracyMode = AccuracyMode.prioritize_long_combinations(),
        fabrication_mode: FabricationMode = FabricationMode.balanced(),
        empty_value: str = "",
        use_synthetic_counts: bool = True,
        aggregate_counts_scale_factor: float = 1.0,
    ) -> None:
        """
        Anonymizes a given dataframe that has been preformatted as categorical microdata (one subject per row; one row per subject).

        See [Synthetic Data Showcase](https://github.com/microsoft/synthetic-data-showcase) for more information.

        Args:
            df (pd.DataFrame): The dataframe to be anonymized.
            epsilon (float): The epsilon value for differential privacy.
            reporting_length (int, optional): The maximum length of attribute value combination to compute. Defaults to 4.
            percentile_percentage (float, optional): The percentile to use for the epsilon budget. Defaults to 99.
            percentile_epsilon_proportion (float, optional): The proportion of the epsilon budget to use for percentile calculation. Defaults to 0.01.
            number_of_records_epsilon_proportion (float, optional): The proportion of the epsilon budget to use for the number of records. Defaults to 0.005.
            weight_selection_percentile (float, optional): The percentile to use for selecting weights. Defaults to 95.
            accuracy_mode (AccuracyMode, optional): The accuracy mode to use. Defaults to AccuracyMode.prioritize_long_combinations().
            fabrication_mode (FabricationMode, optional): The fabrication mode to use. Defaults to FabricationMode.balanced().
            empty_value (str, optional): The value to use for empty cells. Defaults to "".
            use_synthetic_counts (bool, optional): Whether to use synthetic counts in progress to guide sampling. Defaults to True.
            aggregate_counts_scale_factor (float, optional): The scale factor to use for aggregate counts. Defaults to 1.0.
        """
        self.sensitive_df = df_functions.fix_null_ints(df)

        sensitive_dataset = Dataset.from_data_frame(self.sensitive_df)

        params = (
            DpAggregateSeededParametersBuilder()
            .reporting_length(reporting_length)
            .epsilon(epsilon)
            .percentile_percentage(percentile_percentage)
            .percentile_epsilon_proportion(percentile_epsilon_proportion)
            .accuracy_mode(accuracy_mode)
            .number_of_records_epsilon_proportion(number_of_records_epsilon_proportion)
            .fabrication_mode(fabrication_mode)
            .empty_value(empty_value)
            .weight_selection_percentile(weight_selection_percentile)
            .use_synthetic_counts(use_synthetic_counts)
            .aggregate_counts_scale_factor(aggregate_counts_scale_factor)
            .build()
        )

        synth = DpAggregateSeededSynthesizer(params)

        synth.fit(sensitive_dataset)
        self.protected_number_of_records = synth.get_dp_number_of_records()
        self.delta = 1.0 / (
            math.log(self.protected_number_of_records)
            * self.protected_number_of_records
        )
        synthetic_raw_data = synth.sample()
        synthetic_dataset = Dataset(synthetic_raw_data)
        self.synthetic_df = Dataset.raw_data_to_data_frame(synthetic_raw_data)

        sensitive_aggregates = sensitive_dataset.get_aggregates(reporting_length, ";")

        # export the differentially private aggregates (internal to the synthesizer)
        dp_aggregates = synth.get_dp_aggregates(";")

        # generate aggregates from the synthetic data
        synthetic_aggregates = synthetic_dataset.get_aggregates(reporting_length, ";")

        sensitive_aggregates_parsed = {
            tuple(agg.split(";")): count
            for (agg, count) in sensitive_aggregates.items()
        }
        dp_aggregates_parsed = {
            tuple(agg.split(";")): count for (agg, count) in dp_aggregates.items()
        }
        synthetic_aggregates_parsed = {
            tuple(agg.split(";")): count
            for (agg, count) in synthetic_aggregates.items()
        }

        self.aggregate_df = pd.DataFrame(
            data=dp_aggregates.items(),
            columns=["selections", "protected_count"],
        )
        self.aggregate_df.loc[len(self.aggregate_df)] = [
            "record_count",
            self.protected_number_of_records,
        ]
        self.aggregate_df = self.aggregate_df.sort_values(
            by=["protected_count"], ascending=False
        )
        self.aggregate_error_report = ErrorReport(
            sensitive_aggregates_parsed, dp_aggregates_parsed
        ).gen()
        self.synthetic_error_report = ErrorReport(
            sensitive_aggregates_parsed, synthetic_aggregates_parsed
        ).gen()

    def get_data_schema(self) -> dict[list[str]]:
        return queries.get_data_schema(self.synthetic_df)

    def compute_aggregate_graph_df(
        self,
        filters: list[str],
        source_attribute: str,
        target_attribute: str,
        highlight_attribute: str,
    ) -> pd.DataFrame:
        return queries.compute_aggregate_graph(
            self.aggregate_df,
            filters,
            source_attribute,
            target_attribute,
            highlight_attribute,
        )

    def compute_synthetic_graph_df(
        self,
        filters: list[str],
        source_attribute: str,
        target_attribute: str,
        highlight_attribute,
    ) -> pd.DataFrame:
        return queries.compute_synthetic_graph(
            self.synthetic_df,
            filters,
            source_attribute,
            target_attribute,
            highlight_attribute,
        )

    def compute_time_series_query_df(
        self,
        selection,
        time_attribute,
        series_attributes,
        att_separator=";",
        val_separator=":",
    ) -> pd.DataFrame:
        return queries.compute_time_series_query(
            query=selection,
            sdf=self.synthetic_df,
            adf=self.aggregate_df,
            time_attribute=time_attribute,
            time_series=series_attributes,
            att_separator=att_separator,
            val_separator=val_separator,
        )

    def compute_top_attributes_query_df(
        self,
        query: str,
        show_attributes: list[str],
        num_values: int,
        att_separator=";",
        val_separator=":",
    ) -> pd.DataFrame:
        return queries.compute_top_attributes_query(
            query,
            self.synthetic_df,
            self.aggregate_df,
            show_attributes,
            num_values,
            att_separator,
            val_separator,
        )

    def get_bar_chart_fig(
        self,
        selection: list[str],
        show_attributes: list[str],
        unit: str,
        width: int,
        height: int,
        scheme: list[str],
        num_values: int,
        att_separator=";",
        val_separator=":",
    ) -> tuple[go.Figure, pd.DataFrame]:
        chart_df = self.compute_top_attributes_query_df(
            query=selection,
            show_attributes=show_attributes,
            num_values=num_values,
            att_separator=att_separator,
            val_separator=val_separator,
        )
        chart = visuals.get_bar_chart(
            selection=selection,
            show_attributes=show_attributes,
            unit=unit,
            chart_df=chart_df,
            width=width,
            height=height,
            scheme=scheme,
        )
        return chart, chart_df

    def get_line_chart_fig(
        self,
        selection: list[str],
        series_attributes: list[str],
        unit: str,
        time_attribute: str,
        width: int,
        height: int,
        scheme: list[str],
        att_separator: str = ";",
        val_separator: str = ":",
    ) -> tuple[go.Figure, pd.DataFrame]:
        chart_df = self.compute_time_series_query_df(
            selection=selection,
            time_attribute=time_attribute,
            series_attributes=series_attributes,
            att_separator=att_separator,
            val_separator=val_separator,
        )
        chart = visuals.get_line_chart(
            selection=selection,
            series_attributes=series_attributes,
            unit=unit,
            chart_df=chart_df,
            time_attribute=time_attribute,
            width=width,
            height=height,
            scheme=scheme,
        )
        return chart, chart_df

    def get_flow_chart_fig(
        self,
        selection: list[str],
        source_attribute: str,
        target_attribute: str,
        highlight_attribute: str,
        width: int,
        height: int,
        unit: str,
        scheme: list[str],
        att_separator: str = ";",
        val_separator: str = ":",
    ):
        selection_keys = [x["attribute"] + val_separator + x["value"] for x in selection]
        att_count = 2 if highlight_attribute == "" else 3
        att_count += len(selection)
        if att_count <= 4:
            chart_df = queries.compute_aggregate_graph(
                self.aggregate_df,
                selection_keys,
                source_attribute,
                target_attribute,
                highlight_attribute,
                att_separator,
                val_separator,
            )
        else:
            chart_df = queries.compute_synthetic_graph(
                self.synthetic_df,
                selection_keys,
                source_attribute,
                target_attribute,
                highlight_attribute,
                att_separator,
                val_separator,
            )
        chart = visuals.get_flow_chart(
            chart_df,
            selection,
            source_attribute,
            target_attribute,
            highlight_attribute,
            width,
            height,
            unit,
            scheme,
        )
        return chart, chart_df
