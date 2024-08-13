# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import networkx as nx

from toolkit.risk_networks.nodes import get_community_nodes, get_subgraph


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

    def test_max_cluster_size(self):
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
        max_cluster_size = 2
        community_nodes, _ = get_subgraph(
            graph, nodes, max_cluster_size=max_cluster_size
        )

        for community in community_nodes:
            assert len(community) <= max_cluster_size

    def test_max_cluster_size_high(self):
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
        max_cluster_size = 10
        community_nodes, _ = get_subgraph(
            graph, nodes, max_cluster_size=max_cluster_size
        )

        for community in community_nodes:
            assert len(community) <= max_cluster_size

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
