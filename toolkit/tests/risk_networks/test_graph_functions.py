# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import networkx as nx
import pytest

from toolkit.risk_networks.graph_functions import (
    _merge_condition,
    _merge_node_list,
    _merge_nodes,
    simplify_graph,
)


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
        "toolkit.risk_networks.graph_functions._merge_nodes"
    ).return_value = graph

    aba = simplify_graph(graph)
    assert len(aba.nodes()) == 3


def test_simplify_condition_true(mocker, graph):
    G = nx.Graph()
    mocker.patch("toolkit.risk_networks.graph_functions._merge_nodes").return_value = G

    aba = simplify_graph(graph)
    assert len(aba.nodes()) == 0
