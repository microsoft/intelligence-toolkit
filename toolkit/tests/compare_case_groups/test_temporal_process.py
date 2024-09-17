# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pandas as pd
import polars as pl
import pytest

from toolkit.compare_case_groups.temporal_process import (
    build_temporal_count,
    build_temporal_data,
    calculate_window_delta,
    create_window_df,
)


class TestCreateWindowDf:
    def test_basic(self) -> None:
        # Assuming the groups, temporal, and aggregates variables are defined somewhere
        groups = ["group1", "group2"]
        temporal = "time"
        aggregates = ["agg1", "agg2"]

        data = {
            "group1": ["A", "A", "B", "B"],
            "group2": ["X", "X", "Y", "Y"],
            "time": [1, 2, 1, 2],
            "agg1": [10, 20, 30, 40],
            "agg2": [5, 15, 25, 35],
        }
        wdf = pl.DataFrame(data)

        # Expected output
        expected_data = {
            "group1": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "group2": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
            "time": [1, 1, 2, 2, 1, 1, 2, 2],
            "Attribute Value": [
                "agg1:10",
                "agg2:5",
                "agg1:20",
                "agg2:15",
                "agg1:30",
                "agg2:25",
                "agg1:40",
                "agg2:35",
            ],
            "time Window Count": [1, 1, 1, 1, 1, 1, 1, 1],
        }
        expected_df = pl.DataFrame(expected_data).sort(
            [*groups, temporal, "Attribute Value"]
        )

        # Call the function with the sample DataFrame
        result_df = create_window_df(groups, temporal, aggregates, wdf)

        # Assert the result
        assert result_df.equals(expected_df)

class TestCalculateWindowDelta:
    def test_basic(self) -> None:
        data = {
            "Group": ["A", "A", "B", "B"],
            "Temporal": [1, 2, 1, 2],
            "Attribute Value": ["X:10", "X:20", "Y:30", "Y:40"],
            "Temporal Window Count": [5, 3, 8, 6],
        }
        sample_data = pl.DataFrame(data)
        temporal = "Temporal"

        result = calculate_window_delta(sample_data, temporal)

        assert "Temporal Window Delta" in result.columns
        assert all(result["Temporal Window Delta"].is_not_nan())

    def test_missing_values(self):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Temporal": [1, 2, 1, 2],
            "Attribute Value": ["X:10", "X:20", "Y:30", "Y:40"],
            "Temporal Window Count": [5, None, 8, None],
        }
        sample_data = pl.DataFrame(data)
        temporal = "Temporal"

        result = calculate_window_delta(sample_data, temporal)

        assert result["Temporal Window Count"].is_nan().sum() == 0
        assert result.filter(pl.col("Temporal Window Delta") == 0).height == 4

class TestBuildTemporalCount:
    def test_basic(self) -> None:
        data = {
            "Group": ["A", "A", "B", "B"],
            "Temporal": [1, 2, 1, 2],
            "Attribute Value": ["X:10", "X:20", "Y:30", "Y:40"],
            "Temporal Window Count": [5, 3, 8, 6],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "Temporal"

        result = build_temporal_count(sample_data, groups, temporal)

        assert "Temporal Window Delta" in result.columns
        assert all(result["Temporal Window Delta"].is_not_nan())

    def test_missing_values(self):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Temporal": [1, 2, 1, 20],
            "Attribute Value": ["X:10", "X:20", "Y:30", "Y:40"],
            "Temporal Window Count": [5, None, 8, None],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "Temporal"

        result = build_temporal_count(sample_data, groups, temporal)

        assert result["Temporal Window Count"].is_nan().sum() == 0

    def test_multiple_groups(self):
        data = {
            "Group": ["A", "A", "B", "B"],
            "SubGroup": ["X", "Y", "X", "Y"],
            "Temporal": [1, 2, 1, 2],
            "Attribute Value": ["V1:10", "V2:20", "V1:30", "V2:40"],
            "Temporal Window Count": [5, 3, 8, 6],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group", "SubGroup"]
        temporal = "Temporal"

        result = build_temporal_count(sample_data, groups, temporal)

        # get how many unique groups in result.groupby(["Group", "SubGroup"])
        n_groups = result.group_by(["Group", "SubGroup"]).agg(pl.len()).height

        assert n_groups == 4

    def test_delta_calculation(self):
        data = {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "Temporal": [1, 2, 3, 1, 2, 3],
            "Attribute Value": ["X:10", "X:20", "X:30", "Y:40", "Y:50", "Y:60"],
            "Temporal Window Count": [5, 3, 1, 8, 6, 4],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "Temporal"
        result = build_temporal_count(sample_data, groups, temporal)

        group_a_deltas = result.filter(pl.col("Group") == "A").select(
            "Temporal Window Delta"
        )
        group_a_deltas_values = (
            group_a_deltas.filter(pl.col("Temporal Window Delta") != 0.0)
            .to_series()
            .to_list()
        )

        group_b_deltas = result.filter(pl.col("Group") == "B").select(
            "Temporal Window Delta"
        )
        group_b_deltas_values = (
            group_b_deltas.filter(pl.col("Temporal Window Delta") != 0.0)
            .to_series()
            .to_list()
        )

        # Assertions
        assert group_a_deltas.height == 21
        assert group_b_deltas.height == 21
        for v in [-2, -2]:
            assert v in group_a_deltas_values
        for v in [7, -2, -2]:
            assert v in group_b_deltas_values

        assert (
            result.select(pl.col("Temporal Window Delta").is_not_nan())
            .to_series()
            .all()
        )

        data = {
            "Group": ["A", "A", "B", "B"],
            "Temporal": [1, 2, 1, 2],
            "Attribute Value": ["X:10", "X:20", "Y:30", "Y:40"],
            "Temporal Window Count": [5, 3, 8, 6],
        }


class TestBuildTemporalData:
    @pytest.fixture()
    def expected_df_mock(self):
        return pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
                "Attribute Value": ["V1:10", "V2:20", "V1:30", "V2:40"],
                "Temporal Window Count": [5, 3, 8, 6],
            }
        )

    def test_empty_dataframe(self):
        ldf = pl.DataFrame()
        result = build_temporal_data(ldf, groups=[], temporal_atts=[], temporal="")
        assert result.is_empty()

    def test_single_temporal_attribute(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": ["V1:10", "V2:20", "V1:30", "V2:40"],
            "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf, groups=["Group"], temporal_atts=["2023-01-01"], temporal="Temporal"
        )
        expected_data = {
            "Group": ["A", "B"],
            "Temporal": ["2023-01-01", "2023-01-01"],
            "Attribute Value": ["V1:10", "V1:30"],
            "Temporal Window Count": [5, 8],
            "Temporal Window Rank": [1.0, 1.0],
        }
        expected_df = pl.DataFrame(expected_data)
        assert result.equals(expected_df)

    def test_multiple_temporal_attributes(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": [1, 2, 3, 4],
            "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="Temporal",
        )
        expected_data = {
            "Group": ["A", "B", "A", "B"],
            "Temporal": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "Attribute Value": ["V1:10", "V1:30", "V2:20", "V2:40"],
            "Temporal Window Count": [5, 8, 3, 6],
            "Temporal Window Rank": [1.0, 1.0, 1.0, 1.0],
        }
        expected_df = pl.DataFrame(expected_data)
        assert result.equals(expected_df)

    def test_multiple_groups(self, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "SubGroup": ["X", "Y", "X", "Y"],
            "Attribute Value": [1, 2, 3, 4],
            "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        df_mock = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "SubGroup": ["X", "Y", "X", "Y"],
                "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
                "Attribute Value": ["V1:10", "V2:20", "V1:30", "V2:40"],
                "Temporal Window Count": [5, 3, 8, 6],
            }
        )
        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group", "SubGroup"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="Temporal",
        )
        expected_data = {
            "Group": ["A", "B", "A", "B"],
            "SubGroup": ["X", "X", "Y", "Y"],
            "Temporal": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "Attribute Value": ["V1:10", "V1:30", "V2:20", "V2:40"],
            "Temporal Window Count": [5, 8, 3, 6],
            "Temporal Window Rank": [1.0, 1.0, 1.0, 1.0],
        }
        expected_df = pl.DataFrame(expected_data)
        assert result.equals(expected_df)

    def test_missing_values(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": [1, 2, 3, 4],
            "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="Temporal",
        )

        assert result["Temporal Window Count"].is_nan().sum() == 0

    def test_non_existent_temporal_values(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": [1, 2, 3, 4],
            "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group"],
            temporal_atts=["2023-01-03"],
            temporal="Temporal",
        )

        assert result.is_empty()

    def test_incorrect_groups(self, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": [1, 2, 3, 4],
            "Temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = pl.DataFrame()
        result = build_temporal_data(
            ldf,
            groups=["Group", "NonExistent"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="Temporal",
        )

        assert result.is_empty()