import networkx as nx
import pytest

from python.risk_networks.graph_functions import (
    _merge_condition,
    _merge_node_list,
    _merge_nodes,
    build_undirected_graph,
    simplify_graph,
)

# TODO: Tests with entity_label


@pytest.fixture()
def graph():
    G = nx.Graph()
    G.add_node("A", type="TypeA", flags=1)
    G.add_node("B", type="TypeB", flags=0)
    G.add_node("C", type="TypeC", flags=1)
    G.add_node("D", type="TypeC", flags=1)
    G.add_node("E", type="TypeC", flags=0)
    G.add_node("F", type="TypeC", flags=1)
    G.add_edge("A", "B")
    G.add_edge("B", "F")
    G.add_edge("B", "C")
    G.add_edge("E", "F")
    G.add_edge("D", "F")
    G.add_edge("A", "F")
    return G


def test_merge_node_list(graph):
    merge_list = ["A", "C"]
    merged_graph = _merge_node_list(graph, merge_list)

    assert merged_graph.has_node("A;C")
    assert merged_graph.has_node("B")
    assert merged_graph.nodes["A;C"]["type"] == "TypeA;TypeC"
    assert merged_graph.nodes["A;C"]["flags"] == 1
    assert merged_graph.has_edge("B", "A;C")
    assert not merged_graph.has_node("A")
    assert not merged_graph.has_node("C")
    assert not merged_graph.has_edge("A", "B")
    assert not merged_graph.has_edge("B", "C")
    assert not merged_graph.has_edge("A", "B")
    assert not merged_graph.has_edge("B", "C")
    assert not merged_graph.has_edge("B", "C")


def test_merge_condition():
    x = "A==1;B==2;C==3"
    y = "D==4;E==1;F==5"
    assert _merge_condition(x, y) is True

    # Test case 2: x and y have a common attribute value in the second position
    x = "A==1;B==2;C==3"
    y = "D==4;E==5;F==2"
    assert _merge_condition(x, y) is True

    # Test case 3: x and y do not have any common attribute values
    x = "A==1;B==2;C==3"
    y = "D==4;E==5;F==6"
    assert _merge_condition(x, y) is False

    # Test case 4: x and y have multiple common attribute values
    x = "A==1;B==2;C==3"
    y = "D==4;E==1;F==2"
    assert _merge_condition(x, y) is True


def merge_node_list(G, merge_list):
    # Mock implementation: merges nodes by combining them into a single node.
    # This is a placeholder for the actual merge logic.
    G.remove_nodes_from(merge_list[1:])
    G.add_node(merge_list[0])
    return G


def test_merge_all_nodes(graph):
    merged_graph = _merge_nodes(graph, lambda _x, _y: True)
    assert len(merged_graph.nodes()) < len(graph.nodes())


def test_merge_no_nodes(graph):
    merged_graph = _merge_nodes(graph, lambda _x, _y: False)
    assert len(merged_graph.nodes()) == len(graph.nodes())


def test_empty_graph():
    G = nx.Graph()
    merged_graph = _merge_nodes(G, lambda _x, _y: True)
    assert len(merged_graph.nodes()) == 0


def test_simplify_condition_false(mocker, graph):
    mocker.patch(
        "python.risk_networks.graph_functions._merge_nodes"
    ).return_value = graph

    aba = simplify_graph(graph)
    assert len(aba.nodes()) == 3


def test_simplify_condition_true(mocker, graph):
    G = nx.Graph()
    mocker.patch("python.risk_networks.graph_functions._merge_nodes").return_value = G

    aba = simplify_graph(graph)
    assert len(aba.nodes()) == 0


class TestBuildUndirectedGraph:
    def test_build_undirected_graph_empty(self):
        result = build_undirected_graph()
        assert len(result.nodes()) == 0
        assert len(result.edges()) == 0

    def test_network_attribute_links(self):
        network_attribute_links = [
            [("Entity1", "attribute", "Value1"), ("Entity2", "attribute", "Value2")],
            [("Entity3", "relation", "Entity4"), ("Entity5", "relation", "Entity6")],
            [("Entity7", "attribute", "Value3")],
        ]
        result = build_undirected_graph(network_attribute_links)
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

    def test_network_entity_links(self):
        network_entity_links = [
            ("Entity1", "Parent", "Entity2"),
            ("Entity2", "Parent", "Entity3"),
            ("Entity3", "Parent", "Entity1"),
            ("Entity4", "Parent", "Value1"),
            ("Entity5", "Parent", "Value2"),
            ("Entity1", "Parent", "Value3"),
        ]
        result = build_undirected_graph(network_entity_links=network_entity_links)

        expected_edges = [
            ("ENTITY==Entity1", "ENTITY==Entity2"),
            ("ENTITY==Entity1", "ENTITY==Entity3"),
            ("ENTITY==Entity1", "ENTITY==Value3"),
            ("ENTITY==Entity2", "ENTITY==Entity3"),
            ("ENTITY==Entity4", "ENTITY==Value1"),
            ("ENTITY==Entity5", "ENTITY==Value2"),
        ]
        expected_nodes = [
            "ENTITY==Entity1",
            "ENTITY==Entity2",
            "ENTITY==Entity3",
            "ENTITY==Entity4",
            "ENTITY==Value1",
            "ENTITY==Entity5",
            "ENTITY==Value2",
            "ENTITY==Value3",
        ]

        for node in expected_nodes:
            assert result.has_node(node)
        for edge in expected_edges:
            assert result.has_edge(edge[0], edge[1])
