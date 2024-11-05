# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import shutil
from functools import wraps
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from intelligence_toolkit.anonymize_case_data.api import (
    AnonymizeCaseData,
    SynthesizabilityStatistics,
)
from intelligence_toolkit.anonymize_case_data.visuals import color_schemes
from intelligence_toolkit.helpers import df_functions

example_outputs_folder = "./example_outputs/anonymize_case_data"


def cleanup(skip: bool = False):
    """Decorator to cleanup the output and cache folders after each test."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AssertionError:
                raise
            finally:
                if not skip:
                    root = Path(kwargs["input_path"])
                    shutil.rmtree(root / "anonymize_case_data", ignore_errors=True)

        return wrapper

    return decorator


class TestACD:
    @pytest.fixture()
    def dataset(self):
        data_path = f"{example_outputs_folder}/customer_complaints/customer_complaints_prepared.csv"
        return pd.read_csv(data_path)

    def test_anonymize_case_data(self, dataset):
        acd = AnonymizeCaseData()

        sensitive_data = df_functions.suppress_boolean_binary(dataset)

        assert not sensitive_data.isin([0.0]).any().any()

        synthesizability_stats = acd.analyze_synthesizability(sensitive_data)
        assert synthesizability_stats.num_cols == 9
        assert synthesizability_stats.overall_att_count == 101
        assert synthesizability_stats.possible_combinations == 27648
        assert synthesizability_stats.possible_combinations_per_row == 9.2
        assert synthesizability_stats.mean_vals_per_record == 5.409
        assert round(synthesizability_stats.max_combinations_per_record, 2) == 42.49
        assert round(synthesizability_stats.excess_combinations_ratio, 2) == 0.22

        # Anonymize the data
        acd.anonymize_case_data(
            df=sensitive_data,
            epsilon=12.0,
        )

        assert len(acd.aggregate_df) > 0

        selections = acd.aggregate_df["selections"].to_list()
        assert "age_range:(30-40]" in selections
        assert "record_count" in selections
        assert "quality_issue:True" in selections
        assert "age_range:(40-50];city:Mountainview;period:2023-H1" in selections

        assert "0.00 %" not in acd.aggregate_error_report["Suppressed %"].to_list()

        count_error = acd.aggregate_error_report["Count +/- Error"].to_list()

        assert "160.66" in count_error[0]
        assert "23.85" in count_error[1]
        assert "6.85" in count_error[2]
        assert "2.85" in count_error[3]
        assert "6.88" in count_error[4]

        bar_chart, bar_chart_df = acd.get_bar_chart_fig(
            selection=[],
            show_attributes=[],
            unit="Customer",
            width=700,
            height=400,
            scheme=color_schemes["Alphabet"],
            num_values=10,
        )

        assert isinstance(bar_chart_df, pd.DataFrame), "Expected a pandas DataFrame"
        assert len(bar_chart_df) == 10, "Expected 10 rows in the DataFrame"
        expected_columns = [
            "Attribute",
            "Count",
            "Attribute Value",
        ]
        assert all(
            col in bar_chart_df.columns for col in expected_columns
        ), f"DataFrame should contain columns: {expected_columns}"

        assert bar_chart.layout.width == 700, "Expected bar chart width of 700"
        assert bar_chart.layout.height == 400, "Expected bar chart height of 400"
        assert (
            len(bar_chart.data) > 0
        ), "Expected the chart to contain at least one data trace"
