# # Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# # Licensed under the MIT license. See LICENSE file in the project.
# #

# import polars as pl

# from toolkit.compare_case_groups.temporal_process import create_window_df


# class TestCreateWindowDf:
#     def test_basic(self) -> None:
#         # Assuming the groups, temporal, and aggregates variables are defined somewhere
#         groups = ["group1", "group2"]
#         temporal = "time"
#         aggregates = ["agg1", "agg2"]

#         data = {
#             "group1": ["A", "A", "B", "B"],
#             "group2": ["X", "X", "Y", "Y"],
#             "time": [1, 2, 1, 2],
#             "agg1": [10, 20, 30, 40],
#             "agg2": [5, 15, 25, 35],
#         }
#         wdf = pl.DataFrame(data)

#         # Expected output
#         expected_data = {
#             "group1": ["A", "A", "A", "A", "B", "B", "B", "B"],
#             "group2": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
#             "time": [1, 1, 2, 2, 1, 1, 2, 2],
#             "Attribute Value": [
#                 "agg1:10",
#                 "agg2:5",
#                 "agg1:20",
#                 "agg2:15",
#                 "agg1:30",
#                 "agg2:25",
#                 "agg1:40",
#                 "agg2:35",
#             ],
#             "time Window Count": [1, 1, 1, 1, 1, 1, 1, 1],
#         }
#         expected_df = pl.DataFrame(expected_data).sort(
#             [*groups, temporal, "Attribute Value"]
#         )

#         # Call the function with the sample DataFrame
#         result_df = create_window_df(groups, temporal, aggregates, wdf)

#         # Assert the result
#         assert result_df.equals(expected_df)


import pandas as pd
import polars as pl
import pytest

from toolkit.compare_case_groups.temporal_process import temp_rank


@pytest.fixture()
def sample_data():
    data = {
        "Group": ["A", "A", "A", "B", "B", "B"],
        "Temporal": [1, 2, 3, 1, 2, 3],
        "Attribute Value": ["X", "X", "X", "Y", "Y", "Y"],
        "Temporal Window Count": [5, 3, 1, 8, 6, 4],
    }
    return pd.DataFrame(data)


def test_temp_rank_basic(sample_data):
    groups = ["Group"]
    temporal_atts = ["Temporal"]
    temporal = "Temporal"

    result = temp_rank(sample_data, groups, temporal_atts, temporal)

    assert result.empty


def test_temp_rank_missing_values():
    data = {
        "Group": ["A", "A", "A", "B", "B", "B"],
        "Temporal": [1, 2, 3, 1, 2, 3],
        "Attribute Value": ["X", "X", "X", "Y", "Y", "Y"],
        "Temporal Window Count": [5, None, 1, 8, 6, None],
    }
    sample_data = pd.DataFrame(data)
    groups = ["Group"]
    temporal_atts = ["Temporal"]
    temporal = "Temporal"

    result = temp_rank(sample_data, groups, temporal_atts, temporal)

    assert not result.empty, "The result should not be empty"
    assert (
        result["Temporal Window Count"].isnull().sum() == 0
    ), "There should be no missing values in the result"


def test_temp_rank_ranking(sample_data):
    groups = ["Group"]
    temporal_atts = ["Temporal"]
    temporal = "Temporal"

    result = temp_rank(sample_data, groups, temporal_atts, temporal)

    group_a_ranks = result[result["Group"] == "A"]["Temporal Window Rank"]
    group_b_ranks = result[result["Group"] == "B"]["Temporal Window Rank"]

    assert all(group_a_ranks == [1, 2, 3]), "The ranks for group A are incorrect"
    assert all(group_b_ranks == [1, 2, 3]), "The ranks for group B are incorrect"


def test_temp_rank_delta(sample_data):
    groups = ["Group"]
    temporal_atts = ["Temporal"]
    temporal = "Temporal"

    result = temp_rank(sample_data, groups, temporal_atts, temporal)

    group_a_deltas = result[result["Group"] == "A"]["Temporal Window Delta"]
    group_b_deltas = result[result["Group"] == "B"]["Temporal Window Delta"]

    assert all(group_a_deltas == [0, -2, -2]), "The deltas for group A are incorrect"
    assert all(group_b_deltas == [0, -2, -2]), "The deltas for group B are incorrect"


def test_temp_rank_multiple_groups():
    data = {
        "Group": ["A", "A", "B", "B"],
        "SubGroup": ["X", "Y", "X", "Y"],
        "Temporal": [1, 2, 1, 2],
        "Attribute Value": ["V1", "V2", "V1", "V2"],
        "Temporal Window Count": [5, 3, 8, 6],
    }
    sample_data = pd.DataFrame(data)
    groups = ["Group", "SubGroup"]
    temporal_atts = ["Temporal"]
    temporal = "Temporal"

    result = temp_rank(sample_data, groups, temporal_atts, temporal)

    assert not result.empty, "The result should not be empty"
    assert (
        result.groupby(["Group", "SubGroup"]).ngroups == 4
    ), "There should be 4 groups in the result"
