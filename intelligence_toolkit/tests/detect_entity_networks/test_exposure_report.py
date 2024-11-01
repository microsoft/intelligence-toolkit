# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from collections import defaultdict

import networkx as nx
import polars as pl
import pytest

from intelligence_toolkit.detect_entity_networks.exposure_report import (
    build_exposure_data,
)


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
                "entity": ["A", "C", "D", "F", "Z", "X"],
                "type": [
                    "flags_numb",
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
                    "flags_numb",
                ],
                "count": [8, 2, 3, 0, 3, 2],
                "qualified_entity": [
                    "ENTITY==A",
                    "ENTITY==C",
                    "ENTITY==D",
                    "ENTITY==F",
                    "ENTITY==Z",
                    "ENTITY==X",
                ],
            }
        )

    @pytest.fixture()
    def c_nodes(self):
        return ["ENTITY==A", "ENTITY==C", "ENTITY==D"]

    def test_generation_paths_summary(self, graph, flags, c_nodes):
        summary, paths, _ = build_exposure_data(flags, c_nodes, "C", graph)

        expected_paths = [
            [["ENTITY==D"], ["ENTITY==A"], ["ENTITY==C"]],
            [["ENTITY==D"], ["ENTITY==F"], ["ENTITY==C"]],
            [["ENTITY==A"], ["ENTITY==C"]],
        ]
        expected_summary = {"direct": 2, "indirect": 11, "paths": 3, "entities": 2}

        assert len(paths) == len(expected_paths)
        assert summary == expected_summary
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

    def test_generation_nodes_inferred_not_passed(self, graph, flags, c_nodes):
        graph.add_edge("ENTITY==X", "ENTITY==F")
        _, _, nodes = build_exposure_data(flags, c_nodes, "X", graph)

        expected_nodes = [
            {"node": "ENTITY==C", "flags": 2},
            {"node": "ENTITY==D", "flags": 3},
        ]
        for ex in expected_nodes:
            assert ex in nodes

    def test_generation_nodes_inferred_passed(self, graph, flags, c_nodes):
        graph.add_edge("ENTITY==X", "ENTITY==F")
        inferred_links = defaultdict(set)
        inferred_links["ENTITY==X"].add("ENTITY==F")
        _, _, nodes = build_exposure_data(flags, c_nodes, "X", graph, inferred_links)

        expected_nodes = [
            {"node": "ENTITY==C", "flags": 2},
            {"node": "ENTITY==D", "flags": 3},
            {"node": "ENTITY==X", "flags": 2},
        ]
        for ex in expected_nodes:
            assert ex in nodes

    def test_generation_nodes_inferred_c_nodes_set(self, graph, flags, c_nodes) -> None:
        # c_nodes as set
        c_nodes = {"ENTITY==A", "ENTITY==C", "ENTITY==D"}
        graph.add_edge("ENTITY==X", "ENTITY==F")
        inferred_links = defaultdict(set)
        inferred_links["ENTITY==X"].add("ENTITY==F")
        _, _, nodes = build_exposure_data(flags, c_nodes, "X", graph, inferred_links)

        expected_nodes = [
            {"node": "ENTITY==C", "flags": 2},
            {"node": "ENTITY==D", "flags": 3},
            {"node": "ENTITY==X", "flags": 2},
        ]
        for ex in expected_nodes:
            assert ex in nodes

    def test_generation_nodes_inferred_path(self, graph, flags, c_nodes) -> None:
        graph.add_edge("ENTITY==X", "ENTITY==F")
        inferred_links = defaultdict(set)
        inferred_links["ENTITY==X"].add("ENTITY==F")
        _, paths, nodes = build_exposure_data(
            flags, c_nodes, "F", graph, inferred_links
        )

        expected_nodes = [
            {"node": "ENTITY==C", "flags": 2},
            {"node": "ENTITY==D", "flags": 3},
            {"node": "ENTITY==X", "flags": 2},
        ]
        for ex in expected_nodes:
            assert ex in nodes

        assert [["ENTITY==A"], ["ENTITY==C"], ["ENTITY==F"]] in paths
        assert [["ENTITY==A"], ["ENTITY==D"], ["ENTITY==F"]] in paths
        assert [["ENTITY==C", "ENTITY==D", "ENTITY==X"], ["ENTITY==F"]] in paths
