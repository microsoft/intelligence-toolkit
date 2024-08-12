# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import networkx as nx
import pytest

from toolkit.risk_networks.identify import (
    get_entity_neighbors,
    neighbor_is_valid,
    project_entity_graph,
    trim_nodeset,
)


class TestTrimNodeset:
    @pytest.fixture()
    def overall_graph(self):
        G = nx.Graph()
        # Adding more nodes and edges to the graph
        for i in range(1, 10):
            G.add_node(
                f"Entity{i}", type=f"Type{chr(65 + (i % 3))}"
            )  # Types will be TypeA, TypeB, TypeC
        for i in range(1, 10, 2):
            G.add_edge(f"Entity{i}", f"Entity{i + 1}")
            G.add_edge(f"Entity{i}", f"Entity{i + 2}")
        return G

    def test_trim_nodeset_additional_empty(self, overall_graph):
        max_attribute_degree = 1
        additional_trimmed_attributes = set()
        (trimmed_degrees, trimmed_nodes) = trim_nodeset(
            overall_graph, additional_trimmed_attributes, max_attribute_degree
        )

        trimmed_nodes_expected = {
            "Entity1",
            "Entity3",
            "Entity5",
            "Entity7",
            "Entity9",
        }
        trimmed_degrees_expected = {
            ("Entity1", 2),
            ("Entity3", 3),
            ("Entity5", 3),
            ("Entity7", 3),
            ("Entity9", 3),
        }

        assert trimmed_nodes == trimmed_nodes_expected
        assert trimmed_degrees == trimmed_degrees_expected

    def test_trim_nodeset_additional(self, overall_graph):
        max_attribute_degree = 1
        additional_trimmed_attributes = {"Entity2"}
        (trimmed_degrees, trimmed_nodes) = trim_nodeset(
            overall_graph, additional_trimmed_attributes, max_attribute_degree
        )

        trimmed_nodes_expected = {
            "Entity1",
            "Entity2",
            "Entity3",
            "Entity5",
            "Entity7",
            "Entity9",
        }

        trimmed_degrees_expected = {
            ("Entity1", 2),
            ("Entity3", 3),
            ("Entity5", 3),
            ("Entity7", 3),
            ("Entity9", 3),
        }

        assert trimmed_nodes == trimmed_nodes_expected
        assert trimmed_degrees == trimmed_degrees_expected


class TestProjectEntityGraph:
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

        G.add_edge("ENTITY==1", "ENTITY==2")
        G.add_edge("Attr==Type1", "ENTITY==1")
        G.add_edge("ENTITY==3", "Attr==Type1")
        G.add_edge("ENTITY==3", "AttributeABCD==Type35")
        G.add_edge("ENTITY==3", "ENTITY==5")
        return G

    @pytest.fixture()
    def trimmed_nodeset(self):
        return {"ENTITY==5"}

    @pytest.fixture()
    def inferred_links(self):
        expected_links = defaultdict(set)
        # Inferred links vira um link entre eles
        expected_links["ENTITY==1"].add("ENTITY==5")
        return expected_links

    @pytest.fixture()
    def inferred_links_empty(self):
        return defaultdict(set)

    @pytest.fixture()
    def supporting_attribute_types(self):
        # Liga um node a um edge pelo atributo, se os dois estao ligados ao mesmo atributo
        return ["Attr"]  # If here, aparece em edges e nodes! Se nao, nao

    def test_empty_graph(
        self, trimmed_nodeset, inferred_links, supporting_attribute_types
    ):
        empty_graph = nx.Graph()
        projected = project_entity_graph(
            empty_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        assert len(projected.nodes()) == 0
        assert len(projected.edges()) == 0

    def test_edges_no_inferred(
        self,
        simple_graph,
        trimmed_nodeset,
        inferred_links_empty,
        supporting_attribute_types,
    ):
        projected = project_entity_graph(
            simple_graph,
            trimmed_nodeset,
            inferred_links_empty,
            supporting_attribute_types,
        )
        assert ("ENTITY==1", "ENTITY==2") in projected.edges()
        assert ("ENTITY==5", "ENTITY==3") in projected.edges()

    def test_edges_no_trimmed(
        self,
        simple_graph,
        inferred_links_empty,
        supporting_attribute_types,
    ):
        projected = project_entity_graph(
            simple_graph,
            [],
            inferred_links_empty,
            supporting_attribute_types,
        )
        assert ("ENTITY==1", "ENTITY==2") in projected.edges()
        assert ("ENTITY==3", "ENTITY==5") in projected.edges()

    def test_edges_inferred_trimmed(
        self, simple_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
    ):
        projected = project_entity_graph(
            simple_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        assert ("ENTITY==1", "ENTITY==2") in projected.edges()
        assert ("ENTITY==1", "ENTITY==5") not in projected.edges()
        assert ("ENTITY==5", "ENTITY==3") in projected.edges()

    def test_edges_inferred_no_trimmed(
        self, simple_graph, inferred_links, supporting_attribute_types
    ):
        projected = project_entity_graph(
            simple_graph, [], inferred_links, supporting_attribute_types
        )
        assert ("ENTITY==1", "ENTITY==2") in projected.edges()
        assert ("ENTITY==1", "ENTITY==5") in projected.edges()
        assert ("ENTITY==5", "ENTITY==3") in projected.edges()

    def test_nodes_no_inferred(
        self,
        simple_graph,
        trimmed_nodeset,
        inferred_links_empty,
        supporting_attribute_types,
    ):
        projected = project_entity_graph(
            simple_graph,
            trimmed_nodeset,
            inferred_links_empty,
            supporting_attribute_types,
        )
        assert ("ENTITY==1") in projected.nodes()
        assert ("ENTITY==2") in projected.nodes()
        assert ("ENTITY==3") in projected.nodes()
        assert ("ENTITY==4") not in projected.nodes()

    def test_nodes_inferred(
        self, simple_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
    ):
        projected = project_entity_graph(
            simple_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        assert ("ENTITY==1") in projected.nodes()
        assert ("ENTITY==2") in projected.nodes()
        assert ("ENTITY==3") in projected.nodes()
        assert ("ENTITY==5") in projected.nodes()
        assert ("ENTITY==4") not in projected.nodes()


class TestValidNeighbor:
    @pytest.fixture()
    def supporting_attribute_types(self):
        return ["Node1"]

    @pytest.fixture()
    def trimmed_nodeset(self):
        return ["Node2==1"]

    @pytest.fixture()
    def node1(self):
        return "Node1==2"

    @pytest.fixture()
    def node2(self):
        return "Node2==1"

    def test_empty(self):
        result = neighbor_is_valid("", [], [])
        assert result is False

    def test_is_supported(self, node1, supporting_attribute_types):
        result = neighbor_is_valid(node1, supporting_attribute_types, [])
        assert result is False

    def test_is_not_supported(self, node2, supporting_attribute_types):
        result = neighbor_is_valid(node2, supporting_attribute_types, [])
        assert result is True

    def test_is_trimmed(self, node2, supporting_attribute_types, trimmed_nodeset):
        result = neighbor_is_valid(node2, supporting_attribute_types, trimmed_nodeset)
        assert result is False


class TestGetEntityNeighbors:
    @pytest.fixture()
    def graph(self):
        G = nx.Graph()
        G.add_node("node0")
        G.add_node("node1")
        G.add_node("node2")
        G.add_node("node3")
        G.add_node("node4")
        G.add_node("node5")
        G.add_node("node6")
        G.add_edge("node0", "node1")
        G.add_edge("node1", "node2")
        G.add_edge("node1", "node1")
        G.add_edge("node1", "node3")
        G.add_edge("node2", "node3")
        G.add_edge("node3", "node4")
        G.add_edge("node4", "node5")
        return G

    def test_empty_graph(self):
        result = get_entity_neighbors(nx.Graph(), [], [], "")
        assert result == []

    def test_node_not_int_graph(self, graph):
        with pytest.raises(
            ValueError,
            match="Node node7 not in graph",
        ):
            get_entity_neighbors(graph, [], [], "node7")

    def test_no_inferred(self, graph):
        result = get_entity_neighbors(graph, [], [], "node5")
        assert result == ["node4"]

    def test_inferred(self, graph):
        inferred_links = defaultdict(set)
        inferred_links["node5"].add("node2")
        result = get_entity_neighbors(graph, inferred_links, [], "node5")
        assert result == ["node2", "node4"]

    def test_trimmed(self, graph):
        trimmed = ["node2"]
        result = get_entity_neighbors(graph, [], trimmed, "node1")
        assert result == ["node0", "node3"]

    def test_node_equals(self, graph):
        result = get_entity_neighbors(graph, [], [], "node1")
        assert result == ["node0", "node2", "node3"]


def mock_get_entity_neighbors(overall_graph, inferred_links, trimmed_nodeset, node):
    return overall_graph.neighbors(node)


class TestEntityGraph:
    # Test cases
    @pytest.fixture()
    def sample_graph(self):
        G = nx.Graph()
        G.add_nodes_from(
            [
                ("ENTITY==entity_1"),
                ("ENTITY==entity_2"),
                ("ENTITY==entity_3"),
                ("ENTITY==entity_5"),
                ("attr==att_1"),
                ("attr==att_2"),
                ("attr==att_3"),
                ("attr==att_4"),
            ]
        )

        G.add_edges_from(
            [
                ("ENTITY==entity_1", "attr==att_1"),
                ("ENTITY==entity_5", "ENTITY==entity_2"),
                ("ENTITY==entity_1", "attr==att_2"),
                ("ENTITY==entity_2", "attr==att_2"),
                ("ENTITY==entity_2", "attr==att_3"),
                ("ENTITY==entity_3", "attr==att_4"),
            ]
        )
        return G

    def test_project_entity_graph_empty_graph(self):
        G = nx.Graph()
        trimmed_nodeset = set()
        inferred_links = {}
        supporting_attribute_types = []
        P = project_entity_graph(
            G, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        assert len(P.nodes) == 0

    def test_project_entity_graph_basic(self, sample_graph):
        trimmed_nodeset = set()
        inferred_links = {}
        supporting_attribute_types = []

        P = project_entity_graph(
            sample_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        expected = [
            ("ENTITY==entity_1", "ENTITY==entity_2"),
            ("ENTITY==entity_2", "ENTITY==entity_5"),
        ]
        assert list(P.edges()) == expected

    def test_project_entity_graph_trimmed(self, sample_graph):
        trimmed_nodeset = set()
        trimmed_nodeset.add("ENTITY==entity_5")
        inferred_links = {}
        supporting_attribute_types = []

        P = project_entity_graph(
            sample_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        expected = [
            ("ENTITY==entity_1", "ENTITY==entity_2"),
            ("ENTITY==entity_2", "ENTITY==entity_5"),
        ]
        assert list(P.edges()) == expected

    def test_project_entity_graph_inferred(self, sample_graph):
        trimmed_nodeset = set()
        inferred_links = defaultdict(set)
        inferred_links["ENTITY==entity_1"].add("ENTITY==entity_3")
        supporting_attribute_types = []

        P = project_entity_graph(
            sample_graph, trimmed_nodeset, inferred_links, supporting_attribute_types
        )
        expected = [
            ("ENTITY==entity_1", "ENTITY==entity_3"),
            ("ENTITY==entity_1", "ENTITY==entity_2"),
            ("ENTITY==entity_2", "ENTITY==entity_5"),
        ]
        assert list(P.edges()) == expected
