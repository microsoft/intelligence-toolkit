# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl
import pytest

from toolkit.risk_networks.config import FlagAggregatorType
from toolkit.risk_networks.flags import (
    build_exposure_data,
    build_flags,
    integrate_flags,
    prepare_links,
)


class TestIntegrateFlags:
    @pytest.fixture()
    def graph(self):
        G = nx.Graph()
        G.add_node("A")

        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_node("E")
        G.add_node("F")
        return G

    def test_empty_graph(self):
        result = integrate_flags(nx.Graph(), pl.DataFrame())

        assert len(result.nodes()) == 0

    def test_empty_flags(self, graph):
        result = integrate_flags(graph, pl.DataFrame())

        assert len(result.nodes()) == 0

    def test_integration(self, graph):
        flags = pl.DataFrame(
            {
                "qualified_entity": ["A", "C", "D", "F"],
                "count": [1, 2, 3, 0],
            }
        )

        result = integrate_flags(graph, flags)

        assert result.nodes["A"]["flags"] == 1
        assert result.nodes["C"]["flags"] == 2
        assert result.nodes["D"]["flags"] == 3
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]

    def test_sum(self, graph):
        flags = pl.DataFrame(
            {
                "qualified_entity": ["A", "C", "D", "F", "A"],
                "count": [1, 2, 3, 0, 5],
            }
        )

        result = integrate_flags(graph, flags)

        assert result.nodes["A"]["flags"] == 6
        assert result.nodes["C"]["flags"] == 2
        assert result.nodes["D"]["flags"] == 3
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]

    def test_node_not_in_graph(self, graph):
        flags = pl.DataFrame(
            {
                "qualified_entity": ["A", "C", "D", "F", "Z"],
                "count": [1, 2, 3, 0, 3],
            }
        )

        result = integrate_flags(graph, flags)

        assert result.nodes["A"]["flags"] == 1
        assert result.nodes["C"]["flags"] == 2
        assert result.nodes["D"]["flags"] == 3
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]
        assert "Z" not in result.nodes()


class TestPrepareLinks:
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
        flag_agg = FlagAggregatorType.Count.value
        flag_columns = ["flags_numb"]

        result = prepare_links(df_flag, entity_col, flag_agg, flag_columns)

        expected = [
            [
                ["A", "flags_numb", "flags_numb", 1],
                ["C", "flags_numb", "flags_numb", 2],
                ["D", "flags_numb", "flags_numb", 3],
                ["F", "flags_numb", "flags_numb", 0],
                ["Z", "flags_numb", "flags_numb", 3],
            ]
        ]

        assert sorted(result[0]) == sorted(expected[0])

    def test_prepare_instance(self, df_flag):
        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Instance.value
        flag_columns = ["flags_numb"]

        result = prepare_links(df_flag, entity_col, flag_agg, flag_columns)

        expected = [
            [
                ["A", "flags_numb", 1, 1],
                ["C", "flags_numb", 2, 1],
                ["D", "flags_numb", 3, 1],
                ["F", "flags_numb", 0, 1],
                ["Z", "flags_numb", 3, 1],
            ]
        ]

        assert sorted(result[0]) == sorted(expected[0])

    def test_prepare_instance_agg(self, df_flag):
        # add one row to the dataframe
        df_flag = pl.concat(
            [df_flag, pl.DataFrame({"Entity_N": ["A"], "flags_numb": [2]})]
        )

        entity_col = "Entity_N"
        flag_agg = FlagAggregatorType.Instance.value
        flag_columns = ["flags_numb"]

        result = prepare_links(df_flag, entity_col, flag_agg, flag_columns)

        expected = [
            [
                ["A", "flags_numb", 3, 1],
                ["C", "flags_numb", 2, 1],
                ["D", "flags_numb", 3, 1],
                ["F", "flags_numb", 0, 1],
                ["Z", "flags_numb", 3, 1],
            ]
        ]

        assert sorted(result[0]) == sorted(expected[0])


class TestBuildFlags:
    @pytest.fixture()
    def link_list(self):
        return [
            ["A", "flags_numb", 3, 1],
            ["C", "flags_numb", 2, 1],
            ["D", "flags_numb", 3, 1],
            ["F", "flags_numb", 0, 1],
            ["Z", "flags_numb", 3, 1],
        ]

    @pytest.fixture()
    def link_list_count(self):
        return [
            ["A", "flags_numb", "flags_numb", 3],
            ["C", "flags_numb", "flags_numb", 2],
            ["D", "flags_numb", "flags_numb", 3],
            ["F", "flags_numb", "flags_numb", 0],
            ["Z", "flags_numb", "flags_numb", 3],
        ]

    def test_flags_integrated(self, link_list):
        flags, _, _ = build_flags(link_list)

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

        assert df1_sorted.frame_equal(df2_sorted)

    def test_max_entity_flags_integrated(self, link_list):
        _, max_entity_flags, _ = build_flags(link_list)

        expected = 1

        assert max_entity_flags == expected

    def test_mean_entity_flags_integrated(self, link_list):
        _, _, mean_entity_flags = build_flags(link_list)

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

        df1_sorted = flags.sort(by=["entity"])
        df2_sorted = expected.sort(by=["entity"])

        assert df1_sorted.frame_equal(df2_sorted)

    def test_flags_count_sum(self, link_list_count):
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

        df1_sorted = flags.sort(by=["entity"])
        df2_sorted = expected.sort(by=["entity"])

        assert df1_sorted.frame_equal(df2_sorted)

    def test_max_entity_flags_count_sum(self, link_list_count):
        link_list_count.append(["A", "flags_numb", "flags_numb", 5])
        result = build_flags(link_list_count)

        expected = 8

        assert result[1] == expected

    def test_mean_entity_flags_count_sum(self, link_list_count):
        link_list_count.append(["A", "flags_numb", "flags_numb", 5])
        result = build_flags(link_list_count)

        expected = 4.0

        assert result[2] == expected

    def test_max_entity_flags_count(self, link_list_count):
        result = build_flags(link_list_count)

        expected = 3

        assert result[1] == expected

    def test_mean_entity_flags_count(self, link_list_count):
        result = build_flags(link_list_count)

        expected = 2.75

        assert result[2] == expected


class TestExposureData:
    @pytest.fixture()
    def graph(self):
        G = nx.Graph()
        G.add_node("ENTITY==A")
        G.add_node("ENTITY==C")
        G.add_node("ENTITY==D")
        G.add_node("ENTITY==F")
        G.add_node("ENTITY==Z")

        G.add_edge("ENTITY==A", "ENTITY==C")
        G.add_edge("ENTITY==A", "ENTITY==D")
        G.add_edge("ENTITY==C", "ENTITY==F")
        G.add_edge("ENTITY==D", "ENTITY==F")
        return G

    @pytest.fixture()
    def flags(self):
        return pl.DataFrame(
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

    @pytest.fixture()
    def c_nodes(self):
        return ["ENTITY==A", "ENTITY==C", "ENTITY==D"]

    def test_generation_summary(self, graph, flags, c_nodes):
        summary, _, _ = build_exposure_data(flags, c_nodes, "C", graph)

        expected_summary = {"direct": 2, "indirect": 11, "paths": 3, "entities": 2}

        assert summary == expected_summary

    def test_generation_paths(self, graph, flags, c_nodes):
        _, paths, _ = build_exposure_data(flags, c_nodes, "C", graph)

        expected_paths = [
            [["ENTITY==D"], ["ENTITY==A"], ["ENTITY==C"]],
            [["ENTITY==D"], ["ENTITY==F"], ["ENTITY==C"]],
            [["ENTITY==A"], ["ENTITY==C"]],
        ]

        assert len(paths) == len(expected_paths)
        for ex in expected_paths:
            assert ex in paths

    def test_generation_nodes(self, graph, flags, c_nodes):
        _, _, nodes = build_exposure_data(flags, c_nodes, "C", graph)

        expected_nodes = [
            {"node": "ENTITY==A", "flags": 8},
            {"node": "ENTITY==C", "flags": 2},
            {"node": "ENTITY==D", "flags": 3},
        ]
        assert len(nodes) == len(expected_nodes)
        for ex in expected_nodes:
            assert ex in nodes
