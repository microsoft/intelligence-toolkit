# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import polars as pl
import pytest

from intelligence_toolkit.compare_case_groups.temporal_process import (
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
            "attribute_value": [
                "agg1:10",
                "agg2:5",
                "agg1:20",
                "agg2:15",
                "agg1:30",
                "agg2:25",
                "agg1:40",
                "agg2:35",
            ],
            "time_window_count": [1, 1, 1, 1, 1, 1, 1, 1],
        }
        expected_df = pl.DataFrame(expected_data).sort(
            [*groups, temporal, "attribute_value"]
        )

        # Call the function with the sample DataFrame
        result_df = create_window_df(groups, temporal, aggregates, wdf)

        # Assert the result
        assert result_df.equals(expected_df)


class TestCalculateWindowDelta:
    def test_basic(self) -> None:
        data = {
            "Group": ["A", "A", "B", "B"],
            "temporal": [1, 2, 1, 2],
            "attribute_value": ["X:10", "X:20", "Y:30", "Y:40"],
            "temporal_window_count": [5, 3, 8, 6],
        }
        sample_data = pl.DataFrame(data)
        temporal = "temporal"
        groups = ["Group"]

        result = calculate_window_delta(groups, sample_data, temporal)

        assert all(result["temporal_window_delta"].is_not_nan())

    def test_groups(self) -> None:
        data = {
            "Group": [
                "Bayview",
                "Bayview",
                "Bayview",
                "Bayview",
                "Bakeview",
                "Bayview",
            ],
            "temporal": [1, 1, 2, 3, 3, 4],
            "attribute_value": ["X:15", "X:10", "X:10", "X:10", "X:9", "X:10"],
            "temporal_window_count": [9, 5, 3, 8, 7, 6],
        }
        sample_data = pl.DataFrame(data)
        temporal = "temporal"

        expected = {
            "Group": [
                "Bakeview",
                "Bayview",
                "Bayview",
                "Bayview",
                "Bayview",
                "Bayview",
            ],
            "temporal": [3, 1, 1, 2, 3, 4],
            "attribute_value": [
                "X:9",
                "X:10",
                "X:15",
                "X:10",
                "X:10",
                "X:10",
            ],
            "temporal_window_count": [7, 5, 9, 3, 8, 6],
            "temporal_window_delta": [0, 0, 0, -2, 5, -2],
        }
        sample_df = pl.DataFrame(expected)

        groups = ["Group"]

        result = calculate_window_delta(groups, sample_data, temporal)

        assert result.equals(sample_df)

    def test_multiple_groups_no_temporal(self) -> None:
        data = {
            "Group": [
                "Bayview",
                "Westview",
                "Bayview",
                "Bayview",
                "Bakeview",
                "Bayview",
            ],
            "temporal": [1, 2, 2, 3, 3, 4],
            "attribute_value": ["X:10", "X:10", "X:10", "X:10", "X:9", "X:10"],
            "temporal_window_count": [5, 2, 3, 8, 7, 6],
        }
        sample_data = pl.DataFrame(data)
        temporal = "temporal"

        expected = {
            "Group": [
                "Bakeview",
                "Bayview",
                "Bayview",
                "Bayview",
                "Bayview",
                "Westview",
            ],
            "temporal": [3, 1, 2, 3, 4, 2],
            "attribute_value": [
                "X:9",
                "X:10",
                "X:10",
                "X:10",
                "X:10",
                "X:10",
            ],
            "temporal_window_count": [7, 5, 3, 8, 6, 2],
            "temporal_window_delta": [0, 0, -2, 5, -2, 0],
        }
        sample_df = pl.DataFrame(expected)

        groups = ["Group"]

        result = calculate_window_delta(groups, sample_data, temporal)

        assert result.equals(sample_df)

    def test_missing_values(self):
        data = {
            "Group": ["A", "A", "B", "B"],
            "temporal": [1, 2, 1, 2],
            "attribute_value": ["X:10", "X:20", "Y:30", "Y:40"],
            "temporal_window_count": [5, None, 8, None],
        }
        sample_data = pl.DataFrame(data)
        temporal = "temporal"

        groups = ["Group"]

        result = calculate_window_delta(groups, sample_data, temporal)

        assert result["temporal_window_count"].is_nan().sum() == 0
        assert result.filter(pl.col("temporal_window_delta") == 0).height == 4


class TestBuildtemporalCount:
    def test_basic(self) -> None:
        data = {
            "Group": ["A", "A", "B", "B"],
            "temporal": [1, 2, 1, 2],
            "attribute_value": ["X:10", "X:20", "Y:30", "Y:40"],
            "temporal_window_count": [5, 3, 8, 6],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "temporal"

        result = build_temporal_count(sample_data, groups, temporal)

        assert "temporal_window_delta" in result.columns
        assert all(result["temporal_window_delta"].is_not_nan())

    def test_missing_values(self):
        data = {
            "Group": ["A", "A", "B", "B"],
            "temporal": [1, 2, 1, 20],
            "attribute_value": ["X:10", "X:20", "Y:30", "Y:40"],
            "temporal_window_count": [5, None, 8, None],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "temporal"

        result = build_temporal_count(sample_data, groups, temporal)

        assert result["temporal_window_count"].is_nan().sum() == 0

    def test_multiple_groups(self):
        data = {
            "Group": ["A", "A", "B", "B"],
            "SubGroup": ["X", "Y", "X", "Y"],
            "temporal": [1, 2, 1, 2],
            "attribute_value": ["V1:10", "V2:20", "V1:30", "V2:40"],
            "temporal_window_count": [5, 3, 8, 6],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group", "SubGroup"]
        temporal = "temporal"

        result = build_temporal_count(sample_data, groups, temporal)

        # get how many unique groups in result.groupby(["Group", "SubGroup"])
        n_groups = result.group_by(["Group", "SubGroup"]).agg(pl.len()).height

        assert n_groups == 4

    def test_delta_calculation(self):
        data = {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "temporal": [1, 2, 3, 1, 2, 3],
            "attribute_value": ["X:10", "X:20", "X:30", "Y:40", "Y:50", "Y:60"],
            "temporal_window_count": [5, 3, 1, 8, 6, 4],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "temporal"
        result = build_temporal_count(sample_data, groups, temporal)

        group_a_deltas = result.filter(pl.col("Group") == "A").select(
            "temporal_window_delta"
        )
        group_a_deltas_values = (
            group_a_deltas.filter(pl.col("temporal_window_delta") != 0.0)
            .to_series()
            .to_list()
        )

        group_b_deltas = result.filter(pl.col("Group") == "B").select(
            "temporal_window_delta"
        )
        group_b_deltas_values = (
            group_b_deltas.filter(pl.col("temporal_window_delta") != 0.0)
            .to_series()
            .to_list()
        )

        # Assertions
        assert group_a_deltas.height == 9
        assert group_b_deltas.height == 9
        for v in [-5, 3, -3, 1]:
            assert v in group_a_deltas_values
        for v in [-8, 6, -6, 4]:
            assert v in group_b_deltas_values

        assert (
            result.select(pl.col("temporal_window_delta").is_not_nan())
            .to_series()
            .all()
        )

        data = {
            "Group": ["A", "A", "B", "B"],
            "temporal": [1, 2, 1, 2],
            "attribute_value": ["X:10", "X:20", "Y:30", "Y:40"],
            "temporal_window_count": [5, 3, 8, 6],
        }

    def test_delta_calculation_temporal_zeroed(self) -> None:
        data = {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "temporal": [1, 2, 3, 1, 2, 3],
            "attribute_value": ["X:10", "X:20", "X:30", "Y:40", "Y:50", "Y:60"],
            "temporal_window_count": [5, 0, 1, 8, 0, 4],
        }
        sample_data = pl.DataFrame(data)
        groups = ["Group"]
        temporal = "temporal"
        result = build_temporal_count(sample_data, groups, temporal)

        group_a_deltas = result.filter(pl.col("Group") == "A").select(
            "temporal_window_delta"
        )
        group_a_deltas_values = (
            group_a_deltas.filter(pl.col("temporal_window_delta") != 0.0)
            .to_series()
            .to_list()
        )

        group_b_deltas = result.filter(pl.col("Group") == "B").select(
            "temporal_window_delta"
        )
        group_b_deltas_values = (
            group_b_deltas.filter(pl.col("temporal_window_delta") != 0.0)
            .to_series()
            .to_list()
        )

        # Assertions
        assert group_a_deltas.height == 6
        assert group_b_deltas.height == 6
        for v in [-5, 1]:
            assert v in group_a_deltas_values
        for v in [-8, 4]:
            assert v in group_b_deltas_values

        assert (
            result.select(pl.col("temporal_window_delta").is_not_nan())
            .to_series()
            .all()
        )

        data = {
            "Group": ["A", "A", "B", "B"],
            "temporal": [1, 2, 1, 2],
            "attribute_value": ["X:10", "X:20", "Y:30", "Y:40"],
            "temporal_window_count": [5, 3, 8, 6],
        }


class TestBuildtemporalData:
    @pytest.fixture()
    def expected_df_mock(self):
        return pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
                "attribute_value": ["V1:10", "V2:20", "V1:30", "V2:40"],
                "temporal_window_count": [5, 3, 8, 6],
            }
        )

    def test_empty_dataframe(self):
        ldf = pl.DataFrame()
        result = build_temporal_data(ldf, groups=[], temporal_atts=[], temporal="")
        assert result.is_empty()

    def test_single_temporal_attribute(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": ["V1:10", "V2:20", "V1:30", "V2:40"],
            "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf, groups=["Group"], temporal_atts=["2023-01-01"], temporal="temporal"
        )
        expected_data = {
            "Group": ["A", "B"],
            "temporal": ["2023-01-01", "2023-01-01"],
            "attribute_value": ["V1:10", "V1:30"],
            "temporal_window_count": [5, 8],
            "temporal_window_rank": [1.0, 1.0],
        }
        expected_df = pl.DataFrame(expected_data)
        assert result.equals(expected_df)

    def test_multiple_temporal_attributes(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": [1, 2, 3, 4],
            "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="temporal",
        )
        expected_data = {
            "Group": ["A", "B", "A", "B"],
            "temporal": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "attribute_value": ["V1:10", "V1:30", "V2:20", "V2:40"],
            "temporal_window_count": [5, 8, 3, 6],
            "temporal_window_rank": [1.0, 1.0, 1.0, 1.0],
        }
        expected_df = pl.DataFrame(expected_data)
        assert result.equals(expected_df)

    def test_multiple_groups(self, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "SubGroup": ["X", "Y", "X", "Y"],
            "attribute_value": [1, 2, 3, 4],
            "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        df_mock = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "SubGroup": ["X", "Y", "X", "Y"],
                "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
                "attribute_value": ["V1:10", "V2:20", "V1:30", "V2:40"],
                "temporal_window_count": [5, 3, 8, 6],
            }
        )
        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group", "SubGroup"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="temporal",
        )
        expected_data = {
            "Group": ["A", "B", "A", "B"],
            "SubGroup": ["X", "X", "Y", "Y"],
            "temporal": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "attribute_value": ["V1:10", "V1:30", "V2:20", "V2:40"],
            "temporal_window_count": [5, 8, 3, 6],
            "temporal_window_rank": [1.0, 1.0, 1.0, 1.0],
        }
        expected_df = pl.DataFrame(expected_data)
        assert result.equals(expected_df)

    def test_missing_values(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": [1, 2, 3, 4],
            "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="temporal",
        )

        assert result["temporal_window_count"].is_nan().sum() == 0

    def test_non_existent_temporal_values(self, expected_df_mock, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": [1, 2, 3, 4],
            "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = expected_df_mock
        result = build_temporal_data(
            ldf,
            groups=["Group"],
            temporal_atts=["2023-01-03"],
            temporal="temporal",
        )

        assert result.is_empty()

    def test_incorrect_groups(self, mocker):
        data = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": [1, 2, 3, 4],
            "temporal": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        }
        ldf = pl.DataFrame(data)

        mocker.patch(
            "toolkit.compare_case_groups.temporal_process.build_temporal_count"
        ).return_value = pl.DataFrame()
        result = build_temporal_data(
            ldf,
            groups=["Group", "NonExistent"],
            temporal_atts=["2023-01-01", "2023-01-02"],
            temporal="temporal",
        )

        assert result.is_empty()