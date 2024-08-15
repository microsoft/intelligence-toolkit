# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl
import pytest

from toolkit.risk_networks.network import build_fuzzy_neighbors, generate_final_df


class TestFuzzyNeighbors:
    @pytest.fixture()
    def simple_graph(self):
        G = nx.Graph()
        G.add_node("ENTITY==1")
        G.add_node("ENTITY==2")
        G.add_node("ENTITY==3")
        G.add_node("ENTITY==4")
        G.add_node("ENTITY==5")
        G.add_node("Attr==Type1")
        G.add_node("Attr==Type2")
        G.add_node("Attr==Type35")
        G.add_node("AttributeABCD==Type35")
        G.add_node("AttributeABCD==Type37")
        G.add_node("AttributeABCD==Type38")
        G.add_node("AttributeABCD==Type47")

        G.add_edge("ENTITY==1", "ENTITY==2")
        G.add_edge("Attr==Type1", "ENTITY==1")
        G.add_edge("AttributeABCD==Type37", "Attr==Type1")
        G.add_edge("AttributeABCD==Type47", "Attr==Type2")
        G.add_edge("ENTITY==3", "Attr==Type1")
        G.add_edge("ENTITY==3", "AttributeABCD==Type35")
        G.add_edge("AttributeABCD==Type35", "AttributeABCD==Type38")
        G.add_edge("ENTITY==3", "ENTITY==5")
        return G

    @pytest.fixture()
    def existing_network_graph(self):
        G = nx.Graph()
        G.add_node("ENTITY==47")
        G.add_node("Attr==Type108")
        G.add_node("Attr==Type222")

        G.add_edge("Attr==Type108", "ENTITY==47")
        G.add_edge("Attr==Type108", "Attr==Type222")

        return G

    def test_empty_graph(self):
        result = build_fuzzy_neighbors(
            nx.Graph(), nx.Graph(), "ENTITY==1", set(), dict()
        )
        assert result.nodes == nx.Graph().nodes

    def test_node_inexistent(self, simple_graph):
        with pytest.raises(ValueError, match=f"Node ENTITY==78 not in graph"):
            build_fuzzy_neighbors(simple_graph, nx.Graph(), "ENTITY==78", set(), dict())

    def test_fuzzy_neighbor_basic(self, simple_graph):
        network_graph = nx.Graph()
        att_neighbor = "Attr==Type1"
        trimmed_nodeset = set()
        inferred_links = {}
        result = build_fuzzy_neighbors(
            simple_graph, network_graph, att_neighbor, trimmed_nodeset, inferred_links
        )

        assert list(result.nodes()) == ["AttributeABCD==Type37", "Attr==Type1"]
        assert list(result.edges()) == [("AttributeABCD==Type37", "Attr==Type1")]

    def test_fuzzy_neighbor_inferred(self, simple_graph):
        network_graph = nx.Graph()
        att_neighbor = "Attr==Type1"
        trimmed_nodeset = set()
        inf = set()
        inf.add("AttributeABCD==Type47")
        inferred_links = {"Attr==Type1": inf}
        result = build_fuzzy_neighbors(
            simple_graph, network_graph, att_neighbor, trimmed_nodeset, inferred_links
        )

        assert len(result.edges()) == 2
        assert ("AttributeABCD==Type47", "Attr==Type1") in result.edges()
        assert ("AttributeABCD==Type37", "Attr==Type1") in result.edges()

    def test_fuzzy_neighbor_trimmed(self, simple_graph):
        network_graph = nx.Graph()
        att_neighbor = "AttributeABCD==Type35"
        trimmed_nodeset = set()
        trimmed_nodeset.add("AttributeABCD==Type38")
        inferred_links = {}
        result = build_fuzzy_neighbors(
            simple_graph, network_graph, att_neighbor, trimmed_nodeset, inferred_links
        )

        assert len(result.nodes()) == 0
        assert len(result.edges()) == 0

    def test_fuzzy_neighbor_graph_existent(self, simple_graph, existing_network_graph):
        att_neighbor = "Attr==Type1"
        trimmed_nodeset = set()
        inferred_links = {}
        result = build_fuzzy_neighbors(
            simple_graph,
            existing_network_graph,
            att_neighbor,
            trimmed_nodeset,
            inferred_links,
        )

        assert len(result.edges()) == 3
        assert len(result.nodes()) == 5
        assert ("AttributeABCD==Type37", "Attr==Type1") in result.edges()
        assert ("Attr==Type108", "Attr==Type222") in result.edges()


class TestFinalDf:
    @pytest.fixture()
    def community_nodes(self):
        return [
            ["ENTITY==1", "ENTITY==2", "ENTITY==3"],
            ["ENTITY==4", "ENTITY==5", "ENTITY==6"],
        ]

    @pytest.fixture()
    def integrated_flags(self):
        return pl.DataFrame(
            {
                "qualified_entity": ["ENTITY==1", "ENTITY==2", "ENTITY==3"],
                "count": [1, 0, 3],
            }
        )

    def test_final_integrated(self, community_nodes, integrated_flags):
        result = generate_final_df(community_nodes, integrated_flags)

        expected = [
            ("1", 1, 0, 3, 4, 2, 1.33, 2.0),
            ("2", 0, 0, 3, 4, 2, 1.33, 2.0),
            ("3", 3, 0, 3, 4, 2, 1.33, 2.0),
            ("4", 0, 1, 3, 0, 0, 0.0, 0.0),
            ("5", 0, 1, 3, 0, 0, 0.0, 0.0),
            ("6", 0, 1, 3, 0, 0, 0.0, 0.0),
        ]

        assert result == expected

    def test_final_not_integrated(self, community_nodes):
        integrated_flags = None
        result = generate_final_df(community_nodes, integrated_flags)

        expected = [
            ("1", 0, 0, 3, 0, 0, 0, 0),
            ("2", 0, 0, 3, 0, 0, 0, 0),
            ("3", 0, 0, 3, 0, 0, 0, 0),
            ("4", 0, 1, 3, 0, 0, 0, 0),
            ("5", 0, 1, 3, 0, 0, 0, 0),
            ("6", 0, 1, 3, 0, 0, 0, 0),
        ]

        assert result == expected
