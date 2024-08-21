# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import networkx as nx
import polars as pl
import pytest

from toolkit.risk_networks.identify_networks import (
    build_entity_records,
    get_community_nodes,
    get_entity_neighbors,
    get_integrated_flags,
    get_subgraph,
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
        inferred_links["node12"].add("node127")
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


class TestSubgraph:
    def test_basic_functionality(self):
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        nodes = [1, 2, 3]
        community_nodes, entity_to_community = get_subgraph(graph, nodes)
        assert len(community_nodes) == 1
        assert set(community_nodes[0]) == set(nodes)
        assert entity_to_community == {1: 0, 2: 0, 3: 0}

    def test_disconnected_components(self):
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (3, 4)])
        nodes = [1, 2, 3, 4]
        community_nodes, entity_to_community = get_subgraph(graph, nodes)
        assert len(community_nodes) == 2
        assert set(community_nodes[0]) == {1, 2} or set(community_nodes[0]) == {3, 4}
        assert set(community_nodes[1]) == {1, 2} or set(community_nodes[1]) == {3, 4}
        assert len(entity_to_community) == 4

    def test_max_network_entities(self):
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (1, 2),
                (3, 4),
                (7, 9),
                (8, 90),
                (54, 66),
                (66, 44),
                (66, 1),
                (66, 2),
            ]
        )
        nodes = [1, 2, 3, 4, 66, 54, 89]
        max_network_entities = 2
        community_nodes, _ = get_subgraph(
            graph, nodes, max_network_entities=max_network_entities
        )

        for community in community_nodes:
            assert len(community) <= max_network_entities

    def test_max_network_entities_size_high(self):
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (1, 2),
                (3, 4),
                (7, 9),
                (8, 90),
                (54, 66),
                (66, 44),
                (66, 1),
                (66, 2),
            ]
        )
        nodes = [1, 2, 3, 4, 66, 54, 89]
        max_network_entities = 10
        community_nodes, _ = get_subgraph(
            graph, nodes, max_network_entities=max_network_entities
        )

        for community in community_nodes:
            assert len(community) <= max_network_entities

    def test_graph_with_weights(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1.0)
        graph.add_edge(2, 3, weight=2.0)
        graph.add_edge(3, 4, weight=3.0)
        nodes = [1, 2, 3, 4]
        community_nodes, entity_to_community = get_subgraph(graph, nodes)
        assert len(community_nodes) > 0
        assert len(entity_to_community) == 4

    def test_empty_node_list(self):
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        nodes = []
        community_nodes, entity_to_community = get_subgraph(graph, nodes)
        assert community_nodes == []
        assert entity_to_community == {}

    def test_empty_graph(self):
        graph = nx.Graph()
        nodes = [1, 2, 3]
        community_nodes, entity_to_community = get_subgraph(graph, nodes)
        assert community_nodes == []
        assert entity_to_community == {}


class TestNodes:
    def test_nodes(self):
        G = nx.Graph()
        nx.add_path(G, ["node_A", "node_B", "node_C", "node_E"])
        nx.add_path(G, ["node_V", "node_X"])
        nx.add_path(G, ["node_D", "node_P"])
        nx.add_path(G, ["node_D", "node_C"])

        result = get_community_nodes(G, 10)

        assert result == (
            [
                {
                    "node_A",
                    "node_B",
                    "node_C",
                    "node_D",
                    "node_E",
                    "node_P",
                },
                {
                    "node_V",
                    "node_X",
                },
            ],
            {
                "node_A": 0,
                "node_B": 0,
                "node_C": 0,
                "node_D": 0,
                "node_E": 0,
                "node_P": 0,
                "node_V": 1,
                "node_X": 1,
            },
        )

    def test_max_size(self):
        G = nx.Graph()
        nx.add_path(G, ["node_A", "node_B", "node_C", "node_E"])
        nx.add_path(G, ["node_V", "node_X"])
        nx.add_path(G, ["node_D", "node_P"])
        nx.add_path(G, ["node_D", "node_C"])

        result = get_community_nodes(G, 2)

        expected_communities = [
            {
                "node_B",
                "node_A",
            },
            {
                "node_E",
                "node_C",
            },
            {
                "node_D",
                "node_P",
            },
            {
                "node_V",
                "node_X",
            },
        ]

        expected_entity_to_community = {
            "node_A": 0,
            "node_B": 0,
            "node_C": 1,
            "node_D": 2,
            "node_E": 1,
            "node_P": 2,
            "node_V": 3,
            "node_X": 3,
        }

        result_communities = [set(community) for community in result[0]]

        assert len(result_communities) == len(expected_communities)
        for community in expected_communities:
            assert community in result_communities

        assert result[1] == expected_entity_to_community


class TestIntegratedFlags:
    @pytest.fixture()
    def qualified_entities(self) -> list[str]:
        return ["ENTITY==1", "ENTITY==2", "ENTITY==3"]

    @pytest.fixture()
    def integrated_flags(self, qualified_entities) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "qualified_entity": qualified_entities,
                "count": [1, 0, 3],
            }
        )

    def test_empty_integrated_flags(self) -> None:
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


class TestBuildEntityRecords:
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
        result = build_entity_records(community_nodes, integrated_flags)

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
        result = build_entity_records(community_nodes, integrated_flags)

        expected = [
            ("1", 0, 0, 3, 0, 0, 0, 0),
            ("2", 0, 0, 3, 0, 0, 0, 0),
            ("3", 0, 0, 3, 0, 0, 0, 0),
            ("4", 0, 1, 3, 0, 0, 0, 0),
            ("5", 0, 1, 3, 0, 0, 0, 0),
            ("6", 0, 1, 3, 0, 0, 0, 0),
        ]

        assert result == expected
        assert result == expected
