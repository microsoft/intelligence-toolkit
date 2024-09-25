# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import pandas as pd
import polars as pl
import pytest

from toolkit.compare_case_groups.build_dataframes import (
    build_attribute_df,
    build_ranked_df,
    filter_df,
)


class TestBuildRankedGroups:
    @pytest.fixture()
    def sample_data(self):
        ldf = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "attribute_value": [1, 2, 3, 4],
                "attribute_rank": [1, 2, 1, 2],
                "group_rank": [1, 1, 1, 1],
                "temporal column": ["2021-01", "2021-02", "2021-01", "2021-02"],
                "temporal column_window_rank": [1, 2, 1, 2],
                "temporal column_window_delta": [0, 1, 0, 1],
            }
        )

        gdf = pl.DataFrame({"Group": ["A", "B"], "Global Rank": [1, 2]})

        adf = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "attribute_value": [1, 2, 3, 4],
                "attribute_rank": [1, 2, 1, 2],
                "group_rank": [1, 1, 1, 1],
            }
        )

        return ldf, gdf, adf

    def test_build_ranked_df_temporal_columns(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "temporal column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert "temporal column_window" in result_df.columns
        assert "temporal column_window_rank" in result_df.columns
        assert "temporal column_window_delta" in result_df.columns

    def test_build_ranked_df_temporal(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "temporal column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        expected_values = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": [1, 2, 3, 4],
            "attribute_rank": [1, 2, 1, 2],
            "group_rank": [1, 1, 1, 1],
            "temporal column_window_rank": [1, 2, 1, 2],
            "temporal column_window_delta": [0, 1, 0, 1],
        }

        for col, values in expected_values.items():
            assert result_df[col].to_list() == values

    def test_build_ranked_df_no_temporal_columns(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = ""
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert "temporal column_window" not in result_df.columns
        assert "temporal_column_window_rank" not in result_df.columns
        assert "temporal_column_window_delta" not in result_df.columns

    def test_build_ranked_df_no_temporal(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = ""
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        expected_values = {
            "Group": ["A", "A", "B", "B"],
            "attribute_value": [1, 2, 3, 4],
            "attribute_rank": [1, 2, 1, 2],
            "group_rank": [1, 1, 1, 1],
        }

        for col, values in expected_values.items():
            assert result_df[col].to_list() == values

    def test_build_ranked_df_attribute_rank_type(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "temporal column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert result_df["attribute_rank"].dtype == pl.Int32
        assert result_df["group_rank"].dtype == pl.Int32

    def test_build_ranked_df_sorted(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "temporal column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert result_df.equals(result_df.sort(by=groups))
        assert result_df.equals(result_df.sort(by=groups))


class TestFilterDf:
    @pytest.fixture()
    def dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Group": ["A", "A", "B", "B", "C"],
                "attribute_value": ["X", "B", "BCD", "ABC", "X"],
                "attribute_rank": [1, 2, 1, 2, 1],
                "group_rank": [1, 1, 1, 1, 1],
            }
        )

    def test_filter_empty(self, dataset) -> None:
        result = filter_df(dataset, [])
        assert result.equals(dataset)

    def test_filter_single(self, dataset) -> None:
        result = filter_df(dataset, ["Group:A"])
        expected = dataset[dataset["Group"] == "A"]
        assert result.equals(expected)

    def test_filter_multiple(self, dataset) -> None:
        result = filter_df(dataset, ["Group:A", "attribute_value:X"])
        expected = dataset[
            (dataset["Group"] == "A") & (dataset["attribute_value"] == "X")
        ]
        assert result.equals(expected)

    def test_filter_multiple_attr_inexistent(self, dataset) -> None:
        result = filter_df(dataset, ["Group:A", "attribute_value:F"])
        assert len(result) == 0


class TestBuildAttributeDf:
    @pytest.fixture()
    def dataset_2(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Temporal": [1, 2, 1, 2],
                "Aggregate1": [10, 20, None, 40],
                "Aggregate2": [5, None, 15, 20],
            }
        )

    @pytest.fixture()
    def dataset_3(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Group2": ["AX", "AE", "BZ", "BY"],
                "Aggregate1": [40, 20, 30, 50],
                "Aggregate2": [10, 1, 15, 7],
            }
        )

    @pytest.fixture()
    def expected_dataset_1(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Group": [
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                ],
                "attribute_value": [
                    "Aggregate1:20",
                    "Aggregate1:30",
                    "Aggregate1:40",
                    "Aggregate1:50",
                    "Aggregate2:1",
                    "Aggregate2:10",
                    "Aggregate2:15",
                    "Aggregate2:7",
                    "Aggregate1:20",
                    "Aggregate1:30",
                    "Aggregate1:40",
                    "Aggregate1:50",
                    "Aggregate2:1",
                    "Aggregate2:10",
                    "Aggregate2:15",
                    "Aggregate2:7",
                ],
                "attribute_count": [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
                "attribute_rank": [
                    1.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                ],
            }
        ).sort(by=["Group", "attribute_value"])

    @pytest.fixture()
    def expected_dataset_2(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Group": ["A"] * 6 + ["B"] * 6,
                "attribute_value": [
                    "Aggregate1:10",
                    "Aggregate1:20",
                    "Aggregate1:40",
                    "Aggregate2:15",
                    "Aggregate2:20",
                    "Aggregate2:5",
                    "Aggregate1:10",
                    "Aggregate1:20",
                    "Aggregate1:40",
                    "Aggregate2:15",
                    "Aggregate2:20",
                    "Aggregate2:5",
                ],
                "attribute_count": [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                "attribute_rank": [1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2],
            }
        ).sort(by=["Group", "attribute_value"])

    @pytest.fixture()
    def expected_dataset_3(self) -> pl.DataFrame:
        return (
            pl.DataFrame(
                {
                    "Group": ["A"] * 4 + ["B"] * 4 + ["A"] * 12 + ["B"] * 12,
                    "Group2": [
                        "AE",
                        "AE",
                        "AX",
                        "AX",
                        "BY",
                        "BY",
                        "BZ",
                        "BZ",
                        "AE",
                        "AE",
                        "AE",
                        "AE",
                        "AE",
                        "AE",
                        "AX",
                        "AX",
                        "AX",
                        "AX",
                        "AX",
                        "AX",
                        "BY",
                        "BY",
                        "BY",
                        "BY",
                        "BY",
                        "BY",
                        "BZ",
                        "BZ",
                        "BZ",
                        "BZ",
                        "BZ",
                        "BZ",
                    ],
                    "attribute_value": [
                        "Aggregate1:20",
                        "Aggregate2:1",
                        "Aggregate1:40",
                        "Aggregate2:10",
                        "Aggregate1:50",
                        "Aggregate2:7",
                        "Aggregate1:30",
                        "Aggregate2:15",
                        "Aggregate1:40",
                        "Aggregate2:10",
                        "Aggregate1:50",
                        "Aggregate2:7",
                        "Aggregate1:30",
                        "Aggregate2:15",
                        "Aggregate1:20",
                        "Aggregate2:1",
                        "Aggregate1:50",
                        "Aggregate2:7",
                        "Aggregate1:30",
                        "Aggregate2:15",
                        "Aggregate1:20",
                        "Aggregate2:1",
                        "Aggregate1:40",
                        "Aggregate2:10",
                        "Aggregate1:30",
                        "Aggregate2:15",
                        "Aggregate1:20",
                        "Aggregate2:1",
                        "Aggregate1:40",
                        "Aggregate2:10",
                        "Aggregate1:50",
                        "Aggregate2:7",
                    ],
                    "attribute_count": [1] * 8 + [0] * 24,
                    "attribute_rank": [1.0] * 8 + [4.0] * 24,
                }
            )
            .with_columns(
                [
                    pl.col("attribute_rank").cast(pl.UInt32),
                    pl.col("attribute_count").cast(pl.UInt32),
                ]
            )
            .sort(by=["Group", "Group2", "attribute_value"])
        )

    def test_build_attribute_df(self, expected_dataset_1) -> None:
        # Test Case 1: Basic functionality
        df1 = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Temporal": [1, 2, 1, 2],
                "Aggregate1": [40, 20, 30, 50],
                "Aggregate2": [10, 1, 15, 7],
            }
        )

        result_df1 = build_attribute_df(
            df1, ["Group"], ["Aggregate1", "Aggregate2"]
        ).sort(by=["Group", "attribute_value"])
        result_df1.equals(expected_dataset_1)

    def test_with_missing_values(self, dataset_2, expected_dataset_2):
        result_df2 = build_attribute_df(
            dataset_2,
            ["Group"],
            ["Aggregate1", "Aggregate2"],
        ).sort(by=["Group", "attribute_value"])

        assert result_df2.equals(expected_dataset_2)

    def test_with_additional_group(self, dataset_3, expected_dataset_3):
        result_df3 = build_attribute_df(
            dataset_3,
            ["Group", "Group2"],
            ["Aggregate1", "Aggregate2"],
        ).sort(by=["Group", "Group2", "attribute_value"])

        assert result_df3.equals(expected_dataset_3)
