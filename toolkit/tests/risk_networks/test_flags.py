# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl
import pytest

from toolkit.risk_networks.flags import (
    build_exposure_data,
    get_integrated_flags,
    integrate_flags,
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


class TestIntegratedFlags:
    @pytest.fixture()
    def qualified_entities(self):
        return ["ENTITY==1", "ENTITY==2", "ENTITY==3"]

    @pytest.fixture()
    def integrated_flags(self, qualified_entities):
        return pl.DataFrame(
            {
                "qualified_entity": qualified_entities,
                "count": [1, 0, 3],
            }
        )

    def test_empty_integrated_flags(self):
        integrated_flags = pl.DataFrame()
        entities = []
        result = get_integrated_flags(integrated_flags, entities)
        assert result == (0, 0, 0, 0)

    def test_no_entities(self, integrated_flags):
        entities = []
        result = get_integrated_flags(integrated_flags, entities)
        assert result == (0, 0, 0, 0)

    def test_community_flags(self, integrated_flags, qualified_entities):
        community_flags, _, _, _ = get_integrated_flags(
            integrated_flags, qualified_entities
        )

        assert community_flags == 4

    def test_flagged(self, integrated_flags, qualified_entities):
        _, flagged, _, _ = get_integrated_flags(integrated_flags, qualified_entities)

        assert flagged == 2

    def test_flagged_per_unflagged(self, integrated_flags, qualified_entities):
        _, _, flagged_per_unflagged, _ = get_integrated_flags(
            integrated_flags, qualified_entities
        )

        assert flagged_per_unflagged == 2

    def test_flags_per_entity(self, integrated_flags, qualified_entities):
        _, _, _, flags_per_entity = get_integrated_flags(
            integrated_flags, qualified_entities
        )

        assert flags_per_entity == 1.33
