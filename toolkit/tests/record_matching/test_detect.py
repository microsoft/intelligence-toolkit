# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import re

import numpy as np
import polars as pl
import pytest

from toolkit.record_matching.detect import (
    _calculate_mean_score,
    build_matches,
    build_matches_dataset,
    build_near_map,
    build_nearest_neighbors,
    build_sentence_pair_scores,
    convert_to_sentences,
)


class TestConvertToSentences:
    @pytest.fixture()
    def merged_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ID1": [10, 20, 30, 40, 50],
                "Name": ["A", "B", "C", "D", "E"],
                "VehicleType": [
                    "Hatch 1",
                    "Sedan 1",
                    "Truck 1",
                    "SUV 3",
                    "CyberTruck 3",
                ],
                "VehicleColor": ["Blue", "Red", "Blue", "Black", "Silver"],
                "VehicleYear": ["2021", "2022", "2022", "2023", "2024"],
            }
        )

    def test_df_empty(self) -> None:
        df_empty = pl.DataFrame()
        result = convert_to_sentences(df_empty, [])
        assert len(result) == 0

    def test_skip_empty(self, merged_df) -> None:
        result = convert_to_sentences(merged_df, [])

        assert len(result) == 5
        assert "ID1" in result[0]

    def test_skip(self, merged_df) -> None:
        result = convert_to_sentences(merged_df, ["ID1"])

        for re in result:
            assert "ID1" not in re

    def test_sentence(self, merged_df) -> None:
        result = convert_to_sentences(merged_df)

        assert len(result) == 5
        for re in result:
            assert "ID1:" in re
            assert "NAME:" in re
            assert "VEHICLETYPE:" in re
            assert "VEHICLECOLOR:" in re
            assert "VEHICLEYEAR:" in re

    def test_val_nan(self, merged_df) -> None:
        # add one row with nan value
        merged_df = pl.concat(
            [
                merged_df,
                pl.DataFrame(
                    {
                        "ID1": [60],
                        "Name": ["F"],
                        "VehicleType": ["NAN"],
                        "VehicleColor": ["Blue"],
                        "VehicleYear": ["2021"],
                    }
                ),
            ]
        )
        result = convert_to_sentences(merged_df)

        re = result[-1]
        assert "VEHICLETYPE: ;" in re


class TestBuildNearestNeighbors:
    @pytest.fixture()
    def embeddings(self) -> np.array:
        return np.random.rand(1000, 10)

    def test_neighbors_greater_than_embeddings(self) -> None:
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        expected_msg = (
            "Number of neighbors (50) is greater than number of embeddings (3)"
        )
        escaped_expected_msg = re.escape(expected_msg)
        with pytest.raises(ValueError, match=escaped_expected_msg):
            build_nearest_neighbors(embeddings, 50)

    def test_neighbors_5(self, embeddings) -> None:
        result = build_nearest_neighbors(embeddings, 5)

        assert len(result) == 2
        assert result[0].shape == (1000, 5)

    def test_neighbors_10(self, embeddings) -> None:
        result = build_nearest_neighbors(embeddings, 10)

        assert len(result) == 2
        assert result[0].shape == (1000, 10)


class TestBuildNearMap:
    @pytest.fixture()
    def all_sentences(self) -> list[str]:
        return [
            "ID1: 10; NAME: A; VEHICLETYPE: Hatch 1; VEHICLECOLOR: Blue; VEHICLEYEAR: 2021;",
            "ID1: 20; NAME: B; VEHICLETYPE: Sedan 1; VEHICLECOLOR: Red; VEHICLEYEAR: 2022;",
            "ID1: 30; NAME: C; VEHICLETYPE: Truck 1; VEHICLECOLOR: Blue; VEHICLEYEAR: 2022;",
        ]

    def test_result(self, all_sentences) -> None:
        distances = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.03, 0.8, 0.9]])
        indices = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        result = build_near_map(distances, indices, all_sentences)

        assert len(result) == 2
        assert result == {0: [1, 1, 2, 2], 1: [1, 1]}

    def test_result_max_record(self, all_sentences) -> None:
        distances = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.03, 0.8, 0.9]])
        indices = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        result = build_near_map(distances, indices, all_sentences, 0.1)

        assert len(result) == 2
        assert result == {0: [1, 1, 2, 2], 1: [1, 1, 2, 2]}


class TestBuildSentencePairScores:
    @pytest.fixture()
    def near_map(self) -> dict:
        return {0: [1, 1, 2, 2], 1: [1, 1]}

    @pytest.fixture()
    def merged_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ID1": [10, 20, 30, 40, 50],
                "Entity name": ["A", "B", "C", "D", "E"],
                "VehicleType": [
                    "Hatch 1",
                    "Sedan 1",
                    "Truck 1",
                    "SUV 3",
                    "CyberTruck 3",
                ],
                "VehicleColor": ["Blue", "Red", "Blue", "Black", "Silver"],
                "VehicleYear": ["2021", "2022", "2022", "2023", "2024"],
            }
        )

    def test_empty(self) -> None:
        result = build_sentence_pair_scores({}, pl.DataFrame())
        assert result == []

    def test_build_sentence_pair_scores(self, merged_df) -> None:
        near_map = {0: [1, 1, 2, 2], 1: [1, 1]}

        result = build_sentence_pair_scores(near_map, merged_df)
        expected = [(0, 1, 0), (0, 1, 0), (0, 2, 0), (0, 2, 0), (1, 1, 0), (1, 1, 0)]
        assert result == expected

    def test_single_pair(self, merged_df) -> None:
        near_map = {0: [1]}

        result = build_sentence_pair_scores(near_map, merged_df)
        expected = [(0, 1, 0)]
        assert result == expected

    def test_multiple_pairs_different_keys(self, merged_df) -> None:
        near_map = {0: [1, 2], 1: [2, 3]}

        result = build_sentence_pair_scores(near_map, merged_df)
        expected = [(0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 3, 0)]
        assert result == expected

    def test_no_matches(self, merged_df) -> None:
        near_map = {}

        result = build_sentence_pair_scores(near_map, merged_df)
        expected = []
        assert result == expected


class TestBuildMatches:
    @pytest.fixture()
    def merged_df(self) -> pl.DataFrame:
        data = {
            "Entity name": ["A", "B", "C", "D", "E"],
            "Dataset": ["X", "X", "Y", "Y", "Z"],
        }
        return pl.DataFrame(data)

    @pytest.fixture()
    def sentence_pair_scores(self) -> list[tuple[int, int, float]]:
        return [(0, 1, 0.8), (0, 2, 0.6), (3, 4, 0.7)]

    def test_basic_grouping(self, merged_df, sentence_pair_scores) -> None:
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {"A::X": 0, "B::X": 0}
        assert entity_to_group == expected

    def test_empty_scores(self, merged_df) -> None:
        entity_to_group, _, _ = build_matches([], merged_df)
        assert entity_to_group == {}

    def test_all_below_threshold(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.2), (0, 2, 0.3), (3, 4, 0.1)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        assert entity_to_group == {}

    def test_single_pair_above_threshold(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 0,
            "B::X": 0,
        }
        assert entity_to_group == expected

    def test_overlapping_groups(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8), (1, 2, 0.75), (2, 3, 0.85)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 0,
            "B::X": 0,
            "C::Y": 0,
            "D::Y": 0,
        }
        assert entity_to_group == expected

    def test_non_jaccard_change(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8), (3, 4, 0.7)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df, 0.5)
        expected = {
            "A::X": 0,
            "B::X": 0,
            "D::Y": 1,
            "E::Z": 1,
        }
        assert entity_to_group == expected

    def test_non_overlapping_groups(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8), (3, 4, 0.75)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 0,
            "B::X": 0,
            "D::Y": 1,
            "E::Z": 1,
        }
        assert entity_to_group == expected

    def test_both_entities_in_same_group(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8), (1, 2, 0.7), (2, 0, 0.9)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 0,
            "B::X": 0,
            "C::Y": 0,
        }
        assert entity_to_group == expected

    def test_both_entities_in_different_groups(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8), (2, 3, 0.9), (1, 2, 0.76)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 0,
            "B::X": 0,
            "C::Y": 0,
            "D::Y": 0,
        }
        assert entity_to_group == expected

    def test_one_entity_in_group_other_not(self, merged_df) -> None:
        sentence_pair_scores = [(0, 1, 0.8), (4, 4, 0.9)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 1,
            "B::X": 1,
            "E::Z": 0,
        }
        assert entity_to_group == expected

    def test_similar_names_different_datasets(self, merged_df) -> None:
        sentence_pair_scores = [(0, 4, 0.8)]
        entity_to_group, _, _ = build_matches(sentence_pair_scores, merged_df)
        expected = {
            "A::X": 0,
            "E::Z": 0,
        }
        assert entity_to_group == expected

    def test_matches(self, merged_df, sentence_pair_scores) -> None:
        _, matches, _ = build_matches(sentence_pair_scores, merged_df)
        assert len(matches) == 2

    def test_pair_to_match(self, merged_df, sentence_pair_scores) -> None:
        _, _, pair_to_match = build_matches(sentence_pair_scores, merged_df)
        expected = {("A::X", "B::X"): 0.8}
        assert pair_to_match == expected


class TestCalculateMeanScore:
    @pytest.fixture()
    def pair_to_match(self) -> dict:
        return {("A::X", "B::X"): 0.8, ("C::Y", "D::Y"): 0.7}

    @pytest.fixture()
    def entity_to_group(self) -> dict:
        return {"A::X": 0, "B::X": 0, "C::Y": 1, "D::Y": 1}

    def test_empty(self) -> None:
        result = _calculate_mean_score({}, {})
        assert result == {}

    def test_no_matches(self, entity_to_group) -> None:
        result = _calculate_mean_score({}, entity_to_group)
        assert result == {}

    def test_single_group(self, pair_to_match, entity_to_group) -> None:
        result = _calculate_mean_score(pair_to_match, entity_to_group)
        expected = {0: 0.8, 1: 0.7}
        assert result == expected

    def test_multiple_groups(self, pair_to_match, entity_to_group) -> None:
        pair_to_match[("B::X", "C::Y")] = 0.8
        result = _calculate_mean_score(pair_to_match, entity_to_group)
        expected = {0: 0.8, 1: 0.7}
        assert result == expected

    def test_no_group(self, pair_to_match, entity_to_group) -> None:
        pair_to_match[("B::X", "C::Y")] = 0.8
        del entity_to_group["C::Y"]
        result = _calculate_mean_score(pair_to_match, entity_to_group)
        expected = {0: 0.8}
        assert result == expected

    def test_no_pair(self, pair_to_match, entity_to_group) -> None:
        del pair_to_match[("A::X", "B::X")]
        result = _calculate_mean_score(pair_to_match, entity_to_group)
        expected = {1: 0.7}
        assert result == expected


class TestBuildMatchesDataset:
    @pytest.fixture()
    def merged_df(self) -> pl.DataFrame:
        data = {
            "Entity ID": ["10", "20", "30", "40", "50"],
            "Entity name": ["A", "B", "C", "D", "E"],
            "Group ID": [0, 0, 1, 1, 2],
            "Dataset": ["X", "X", "Y", "Y", "Z"],
        }
        return pl.DataFrame(data)

    @pytest.fixture()
    def pair_to_match(self) -> dict[tuple[str, str], float]:
        return {("A::X", "B::X"): 0.8, ("C::Y", "D::Y"): 0.7}

    @pytest.fixture()
    def entity_to_group(self) -> dict:
        return {"A::X": 0, "B::X": 0, "C::Y": 1, "D::Y": 1}

    def test_empty_df(self, entity_to_group) -> None:
        result = build_matches_dataset(pl.DataFrame(), [], entity_to_group)
        assert result.is_empty()

    def test_empty_pair(self, merged_df, entity_to_group) -> None:
        result = build_matches_dataset(merged_df, {}, entity_to_group)
        assert len(result) == 4
        assert result["Name similarity"].sum() == 0

    def test_empty_group(self, merged_df, pair_to_match) -> None:
        result = build_matches_dataset(merged_df, pair_to_match, {})
        assert len(result) == 4
        assert result["Name similarity"].sum() == 0

    def test_basic_grouping(self, merged_df, pair_to_match, entity_to_group) -> None:
        result = build_matches_dataset(merged_df, pair_to_match, entity_to_group)
        assert len(result) == 4
        assert result["Group ID"].n_unique() == 2
        assert result["Group size"].sum() == 8
        assert result["Name similarity"].sum() == 3.0

    def test_basic_grouping_ordering(
        self, merged_df, pair_to_match, entity_to_group
    ) -> None:
        result = build_matches_dataset(merged_df, pair_to_match, entity_to_group)
        columns_ordered = [
            "Group ID",
            "Group size",
            "Entity name",
            "Dataset",
            "Name similarity",
        ]
        assert result.columns[0] == "Group ID"
        assert result.columns[1] == "Group name"

    # def test_single_group(
    #     self, merged_df, sentence_pair_scores, entity_to_group
    # ) -> None:
    #     entity_to_group["E::Z"] = 2
    #     result = build_matches_dataset(merged_df, sentence_pair_scores, entity_to_group)
    #     assert len(result) == 5
    #     assert result["Group ID"].n_unique() == 3
    #     assert result["Group size"].sum() == 3
    #     assert result["Name similarity"].sum() == 2.1
