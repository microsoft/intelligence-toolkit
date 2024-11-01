# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from typing import Literal
from unittest.mock import patch

import polars as pl
import pytest

from toolkit.detect_entity_networks.classes import FlagAggregatorType
from toolkit.detect_entity_networks.prepare_model import (
    build_flag_links,
    build_flags,
    build_groups,
    build_main_graph,
    clean_text,
    format_data_columns,
    generate_attribute_links,
    transform_entity,
)


class TestCleanText:
    def test_remove_punctuation(self) -> None:
        assert clean_text("Hello, world!") == "Hello world"

    def test_remove_special_characters(self) -> None:
        assert clean_text("Hello, world!") == "Hello world"

    def test_reduce_multiple_spaces_to_single(self) -> None:
        assert clean_text("Hello    world") == "Hello world"

    def test_allow_special_characters(self) -> None:
        assert (
            clean_text("Email me@home.com & bring snacks+")
            == "Email me@homecom & bring snacks+"
        )

    def test_combined_scenarios(self) -> None:
        assert (
            clean_text("Hello,  world! Email me@home.com & bring snacks+")
            == "Hello world Email me@homecom & bring snacks+"
        )


class TestFormatDataColumns:
    def test_multiple_columns(self) -> None:
        initial_df = pl.DataFrame(
            {
                "entity_id": ["123 ", " 456"],
                "name": ["John Doe", "Jane Doe"],
                "email": ["john@doe.com", "jane@doe.com"],
            }
        )
        expected_df = pl.DataFrame(
            {
                "entity_id": ["123", "456"],
                "name": ["John Doe", "Jane Doe"],
                "email": ["john@doecom", "jane@doecom"],
            }
        )
        columns_to_link = ["name", "email"]
        entity_id_column = "entity_id"

        result_df = format_data_columns(initial_df, columns_to_link, entity_id_column)

        assert result_df.equals(expected_df)

    @patch("re.sub")
    def test_empty_dataframe(self, mock_clean_text) -> None:
        # Setup
        mock_clean_text.side_effect = lambda x: x
        initial_df = pl.DataFrame({"entity_id": [], "name": [], "email": []})
        columns_to_link = ["name", "email"]
        entity_id_column = "entity_id"

        # Exercise
        result_df = format_data_columns(initial_df, columns_to_link, entity_id_column)

        assert mock_clean_text.call_count == 0
        assert result_df.equals(initial_df)

    @patch("re.sub")
    def test_special_characters_in_entity_id(self, mock_clean_text) -> None:
        # Setup
        mock_clean_text.side_effect = lambda _x, _y, _z: "cleaned"
        initial_df = pl.DataFrame(
            {
                "entity_id": ["@123!", "#456$"],
                "name": ["John Doe", "Jane Doe"],
            }
        )
        columns_to_link = ["name"]
        entity_id_column = "entity_id"

        result_df = format_data_columns(initial_df, columns_to_link, entity_id_column)

        assert mock_clean_text.call_count == 8  # 4 for entity_id + 4 for name
        for val in result_df[entity_id_column]:
            assert val == "cleaned"


class TestPrepareEntityAttribute:
    @pytest.fixture()
    def data(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "entity_id": [1, 2, 3],
                "attribute1": ["A", "B", "A"],
                "attribute2": ["X", "Y", "X"],
            }
        )

    def test_column_name(self, data) -> None:
        entity_id_column = "entity_id"
        columns_to_link = ["attribute1", "attribute2"]
        entity_links = generate_attribute_links(
            data,
            entity_id_column,
            columns_to_link,
        )
        assert len(entity_links) == 2


class TestBuildUndirectedGraph:
    def test_graph_empty(self) -> None:
        result = build_main_graph()
        assert result.size() == 0

    def test_attribute_links(self) -> None:
        network_attribute_links = [
            [("Entity1", "attribute", "Value1"), ("Entity2", "attribute", "Value2")],
            [("Entity3", "relation", "Entity4"), ("Entity5", "relation", "Entity6")],
            [("Entity7", "attribute", "Value3")],
        ]
        result = build_main_graph(network_attribute_links)
        expected_nodes = [
            "ENTITY==Entity1",
            "attribute==Value1",
            "ENTITY==Entity2",
            "attribute==Value2",
            "ENTITY==Entity3",
            "relation==Entity4",
            "ENTITY==Entity5",
            "relation==Entity6",
            "ENTITY==Entity7",
            "attribute==Value3",
        ]

        expected_edges = [
            ("ENTITY==Entity1", "attribute==Value1"),
            ("ENTITY==Entity2", "attribute==Value2"),
            ("ENTITY==Entity3", "relation==Entity4"),
            ("ENTITY==Entity5", "relation==Entity6"),
            ("ENTITY==Entity7", "attribute==Value3"),
        ]
        for node in expected_nodes:
            assert result.has_node(node)
        for edge in expected_edges:
            assert result.has_edge(edge[0], edge[1])


class TestBuildFlagLinks:
    @pytest.fixture()
    def df_flag(self):
        return pl.DataFrame(
            {
                "Entity_N": ["A", "C", "D", "F", "Z"],
                "flags_numb": [1, 2, 3, 0, 3],
            }
        )

    def test_prepare_count(self, df_flag):
        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Count
        flag_columns = ["flags_numb"]

        result = build_flag_links(df_flag, entity_col, flag_agg, flag_columns)

        expected = [
            ["A", "flags_numb", "flags_numb", 1],
            ["C", "flags_numb", "flags_numb", 2],
            ["D", "flags_numb", "flags_numb", 3],
            ["F", "flags_numb", "flags_numb", 0],
            ["Z", "flags_numb", "flags_numb", 3],
        ]

        assert sorted(result) == sorted(expected)

    def test_prepare_value_column_doesnt_exist(self, df_flag):
        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Count
        flag_columns = ["flags_numb123"]
        msg = "Column flags_numb123 not found in the DataFrame."
        with pytest.raises(ValueError, match=msg):
            build_flag_links(df_flag, entity_col, flag_agg, flag_columns)

    def test_prepare_entity_column_doesnt_exist(self, df_flag):
        entity_col = "Entity_N12"
        flag_agg = FlagAggregatorType.Count
        flag_columns = ["flags_numb"]
        msg = "Column Entity_N12 not found in the DataFrame."
        with pytest.raises(ValueError, match=msg):
            build_flag_links(df_flag, entity_col, flag_agg, flag_columns)

    def test_prepare_count_existing(self, df_flag):
        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Count
        flag_columns = ["flags_numb"]

        existing_flags = [["E", "flags_numb1", "flags_numb1", 2]]
        result = build_flag_links(
            df_flag, entity_col, flag_agg, flag_columns, existing_flags
        )

        expected = [
            ["A", "flags_numb", "flags_numb", 1],
            ["C", "flags_numb", "flags_numb", 2],
            ["D", "flags_numb", "flags_numb", 3],
            ["F", "flags_numb", "flags_numb", 0],
            ["Z", "flags_numb", "flags_numb", 3],
            ["E", "flags_numb1", "flags_numb1", 2],
        ]

        assert sorted(result) == sorted(expected)

    def test_prepare_instance(self, df_flag):
        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Instance
        flag_columns = ["flags_numb"]

        result = build_flag_links(df_flag, entity_col, flag_agg, flag_columns)

        expected = [
            ["A", "flags_numb", 1, 1],
            ["C", "flags_numb", 2, 1],
            ["D", "flags_numb", 3, 1],
            ["F", "flags_numb", 0, 1],
            ["Z", "flags_numb", 3, 1],
        ]

        assert sorted(result) == sorted(expected)

    def test_prepare_instance_agg(self, df_flag):
        # add one row to the dataframe
        df_flag = pl.concat(
            [df_flag, pl.DataFrame({"Entity_N": ["A"], "flags_numb": [2]})]
        )

        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Instance
        flag_columns = ["flags_numb"]

        result = build_flag_links(df_flag, entity_col, flag_agg, flag_columns)

        expected = [
            ["A", "flags_numb", 3, 1],
            ["C", "flags_numb", 2, 1],
            ["D", "flags_numb", 3, 1],
            ["F", "flags_numb", 0, 1],
            ["Z", "flags_numb", 3, 1],
        ]

        assert sorted(result) == sorted(expected)


class TestBuildFlags:
    @pytest.fixture()
    def link_list_integrated(self) -> list[list]:
        return [
            ["A", "flags_numb", 3, 1],
            ["C", "flags_numb", 2, 1],
            ["D", "flags_numb", 3, 1],
            ["F", "flags_numb", 0, 1],
            ["Z", "flags_numb", 3, 1],
        ]

    @pytest.fixture()
    def link_list_count(self) -> list[list]:
        return [
            ["A", "flags_numb", "flags_numb", 3],
            ["C", "flags_numb", "flags_numb", 2],
            ["D", "flags_numb", "flags_numb", 3],
            ["F", "flags_numb", "flags_numb", 0],
            ["Z", "flags_numb", "flags_numb", 3],
        ]

    def test_flags_list_empty(self) -> None:
        flags, max_entity_flags, mean_entity_flags = build_flags()

        expected_flags = pl.DataFrame()
        expected_max_entity_flags = 0
        expected_mean_entity_flags = 0

        assert flags.equals(expected_flags)
        assert max_entity_flags == expected_max_entity_flags
        assert mean_entity_flags == expected_mean_entity_flags

    def test_flags_integrated(self, link_list_integrated) -> None:
        flags, _, _ = build_flags(link_list_integrated)

        expected = pl.DataFrame(
            {
                "entity": ["A", "C", "D", "F", "Z"],
                "type": [
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                ],
                "flag": [3, 2, 3, 0, 3],
                "count": [1, 1, 1, 1, 1],
                "qualified_entity": [
                    "ENTITY==A",
                    "ENTITY==C",
                    "ENTITY==D",
                    "ENTITY==F",
                    "ENTITY==Z",
                ],
            }
        )

        df1_sorted = flags.sort(by=["entity"])
        df2_sorted = expected.sort(by=["entity"])

        assert df1_sorted.equals(df2_sorted)

    def test_max_entity_flags_integrated(self, link_list_integrated) -> None:
        _, max_entity_flags, _ = build_flags(link_list_integrated)

        expected = 1

        assert max_entity_flags == expected

    def test_mean_entity_flags_integrated(self, link_list_integrated) -> None:
        _, _, mean_entity_flags = build_flags(link_list_integrated)

        expected = 1

        assert mean_entity_flags == expected

    def test_flags_count(self, link_list_count):
        flags, _, _ = build_flags(link_list_count)

        expected = pl.DataFrame(
            {
                "entity": ["A", "C", "D", "F", "Z"],
                "type": [
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                ],
                "flag": [
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                ],
                "count": [3, 2, 3, 0, 3],
                "qualified_entity": [
                    "ENTITY==A",
                    "ENTITY==C",
                    "ENTITY==D",
                    "ENTITY==F",
                    "ENTITY==Z",
                ],
            }
        )

        flags_sorted = flags.sort(by=["entity"])

        assert flags_sorted.equals(expected)

    def test_flags_count_sum(self, link_list_count) -> None:
        link_list_count.append(["A", "flags_numb", "flags_numb", 5])
        flags, _, _ = build_flags(link_list_count)

        expected = pl.DataFrame(
            {
                "entity": ["A", "C", "D", "F", "Z"],
                "type": [
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                ],
                "flag": [
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                    "flags_numb",
                ],
                "count": [8, 2, 3, 0, 3],
                "qualified_entity": [
                    "ENTITY==A",
                    "ENTITY==C",
                    "ENTITY==D",
                    "ENTITY==F",
                    "ENTITY==Z",
                ],
            }
        )

        flags_sorted = flags.sort(by=["entity"])

        assert flags_sorted.equals(expected)

    def test_max_entity_flags_count_sum(self, link_list_count) -> None:
        link_list_count.append(["A", "flags_numb", "flags_numb", 5])
        result = build_flags(link_list_count)

        expected = 8

        assert result[1] == expected

    def test_mean_entity_flags_count_sum(self, link_list_count) -> None:
        link_list_count.append(["A", "flags_numb", "flags_numb", 5])
        result = build_flags(link_list_count)

        expected = 4.0

        assert result[2] == expected

    def test_max_entity_flags_count(self, link_list_count) -> None:
        result = build_flags(link_list_count)

        expected = 3

        assert result[1] == expected

    def test_mean_entity_flags_count(self, link_list_count) -> None:
        result = build_flags(link_list_count)

        expected = 2.75

        assert result[2] == expected


class TestTransformEntity:
    def test_transform_entity_basic(self) -> None:
        entity = "12345"
        expected = "ENTITY==12345"
        assert transform_entity(entity) == expected

    def test_transform_entity_empty_string(self) -> None:
        entity = ""
        expected = "ENTITY=="
        assert transform_entity(entity) == expected

    def test_transform_entity_special_characters(self) -> None:
        entity = "@$%^&*()"
        expected = "ENTITY==@$%^&*()"
        assert transform_entity(entity) == expected

    def test_transform_entity_numeric(self) -> None:
        entity = "9876543210"
        expected = "ENTITY==9876543210"
        assert transform_entity(entity) == expected

    def test_transform_entity_whitespace(self) -> None:
        entity = "   "
        expected = "ENTITY==   "
        assert transform_entity(entity) == expected

    def test_transform_entity_none(self) -> None:
        entity = None
        expected = "ENTITY==None"
        assert transform_entity(entity) == expected


class TestBuildGroups:
    @pytest.fixture()
    def df_groups(self):
        return pl.DataFrame(
            {
                "entity_id": ["A", "B", "C", "D"],
                "attribute1": ["X", "Y", "Z", "X"],
                "attribute2": ["X", "Y", "Y", "X"],
            }
        )

    @pytest.fixture()
    def entity_col(self) -> Literal["entity_id"]:
        return "entity_id"

    def test_build_groups(self, df_groups, entity_col) -> None:
        value_cols = ["attribute1"]
        group_links = build_groups(value_cols, df_groups, entity_col)

        expected_group_links = [
            [
                ["A", "attribute1", "X"],
                ["B", "attribute1", "Y"],
                ["C", "attribute1", "Z"],
                ["D", "attribute1", "X"],
            ]
        ]

        assert sorted(group_links) == expected_group_links

    def test_build_groups_existing_groups(self, df_groups, entity_col) -> None:
        value_cols = ["attribute1"]
        existing_groups_links = [
            ["Z", "attribute2", "X"],
        ]
        group_links = build_groups(
            value_cols, df_groups, entity_col, existing_groups_links
        )

        expected_group_links = [
            ["Z", "attribute2", "X"],
            [
                ["A", "attribute1", "X"],
                ["B", "attribute1", "Y"],
                ["C", "attribute1", "Z"],
                ["D", "attribute1", "X"],
            ],
        ]

        assert group_links == expected_group_links

    def test_build_groups_two_columns(self, df_groups, entity_col) -> None:
        value_cols = ["attribute1", "attribute2"]
        group_links = build_groups(value_cols, df_groups, entity_col)

        expected_group_links = [
            [
                ["A", "attribute1", "X"],
                ["B", "attribute1", "Y"],
                ["C", "attribute1", "Z"],
                ["D", "attribute1", "X"],
            ],
            [
                ["A", "attribute2", "X"],
                ["B", "attribute2", "Y"],
                ["C", "attribute2", "Y"],
                ["D", "attribute2", "X"],
            ],
        ]

        assert sorted(group_links[0]) == expected_group_links[0]
        assert sorted(group_links[1]) == expected_group_links[1]

    def test_build_groups_column_empty(self, df_groups, entity_col) -> None:
        value_cols = []
        group_links = build_groups(value_cols, df_groups, entity_col)

        assert group_links == []

    def test_build_groups_df_empty(self, entity_col) -> None:
        value_cols = ["attribute1"]
        df_groups = pl.DataFrame()
        group_links = build_groups(value_cols, df_groups, entity_col)

        assert group_links == []

    def test_build_groups_column_doesnt_exists(self, entity_col, df_groups) -> None:
        value_cols = ["attribute12"]
        with pytest.raises(
            ValueError,
            match="Column attribute12 not found in the DataFrame.",
        ):
            build_groups(value_cols, df_groups, entity_col)

    def test_build_groups_entity_doesnt_exist(self, df_groups) -> None:
        entity_col = "entity123"
        value_cols = ["attribute1"]
        with pytest.raises(
            ValueError,
            match="Column entity123 not found in the DataFrame.",
        ):
            build_groups(value_cols, df_groups, entity_col)
