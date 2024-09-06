# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl
import pytest

from toolkit.detect_entity_networks.exposure_report import build_exposure_data


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
