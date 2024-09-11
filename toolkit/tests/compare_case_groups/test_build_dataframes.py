# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import pandas as pd
import polars as pl
import pytest

from toolkit.compare_case_groups.build_dataframes import build_ranked_df, filter_df


class TestBuildRankedGroups:
    @pytest.fixture()
    def sample_data(self):
        ldf = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Attribute Value": [1, 2, 3, 4],
                "Attribute Rank": [1, 2, 1, 2],
                "Group Rank": [1, 1, 1, 1],
                "Temporal Column": ["2021-01", "2021-02", "2021-01", "2021-02"],
                "Temporal Column Window Rank": [1, 2, 1, 2],
                "Temporal Column Window Delta": [0, 1, 0, 1],
            }
        )

        gdf = pl.DataFrame({"Group": ["A", "B"], "Global Rank": [1, 2]})

        adf = pl.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
                "Attribute Value": [1, 2, 3, 4],
                "Attribute Rank": [1, 2, 1, 2],
                "Group Rank": [1, 1, 1, 1],
            }
        )

        return ldf, gdf, adf

    def test_build_ranked_df_temporal_columns(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "Temporal Column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert "Temporal Column Window" in result_df.columns
        assert "Temporal Column Window Rank" in result_df.columns
        assert "Temporal Column Window Delta" in result_df.columns

    def test_build_ranked_df_temporal(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "Temporal Column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        expected_values = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": [1, 2, 3, 4],
            "Attribute Rank": [1, 2, 1, 2],
            "Group Rank": [1, 1, 1, 1],
            "Temporal Column Window Rank": [1, 2, 1, 2],
            "Temporal Column Window Delta": [0, 1, 0, 1],
        }

        for col, values in expected_values.items():
            assert result_df[col].to_list() == values

    def test_build_ranked_df_no_temporal_columns(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = ""
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert "Temporal Column Window" not in result_df.columns
        assert "Temporal Column Window Rank" not in result_df.columns
        assert "Temporal Column Window Delta" not in result_df.columns

    def test_build_ranked_df_no_temporal(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = ""
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        expected_values = {
            "Group": ["A", "A", "B", "B"],
            "Attribute Value": [1, 2, 3, 4],
            "Attribute Rank": [1, 2, 1, 2],
            "Group Rank": [1, 1, 1, 1],
        }

        for col, values in expected_values.items():
            assert result_df[col].to_list() == values

    def test_build_ranked_df_attribute_rank_type(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "Temporal Column"
        groups = ["Group"]

        result_df = build_ranked_df(ldf, gdf, adf, temporal, groups)

        assert result_df["Attribute Rank"].dtype == pl.Int32
        assert result_df["Group Rank"].dtype == pl.Int32

    def test_build_ranked_df_sorted(self, sample_data):
        ldf, gdf, adf = sample_data
        temporal = "Temporal Column"
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
                "Attribute Value": ["X", "B", "BCD", "ABC", "X"],
                "Attribute Rank": [1, 2, 1, 2, 1],
                "Group Rank": [1, 1, 1, 1, 1],
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
        result = filter_df(dataset, ["Group:A", "Attribute Value:X"])
        expected = dataset[
            (dataset["Group"] == "A") & (dataset["Attribute Value"] == "X")
        ]
        assert result.equals(expected)

    def test_filter_multiple_attr_inexistent(self, dataset) -> None:
        result = filter_df(dataset, ["Group:A", "Attribute Value:F"])
        assert len(result) == 0
