# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import networkx as nx
import pytest
from itk_networks.explore import (
    merge_nodes,
    network_subgraph,
    should_merge,
    simplify,
    subgraph_frames,
)
from itk_networks.prepare import (
    attribute_links,
    build_graph,
    flag_links,
    integrate_flags,
)
from itk_networks_testing import A, E, attribute_table, flag_table


@pytest.fixture()
def graph():
    return build_graph(attribute_links(attribute_table()))


@pytest.fixture()
def flags():
    return integrate_flags(flag_links(flag_table()))[0]


def test_subgraph_contains_entities_attributes_and_flags(graph, flags):
    trimmed = frozenset({A("city", "Metro")})
    sub = network_subgraph(
        graph, [E("A"), E("B")], {E("A"): 0, E("B"): 0}, flags, trimmed
    )
    assert set(sub.nodes()) == {
        E("A"),
        E("B"),
        A("phone", "111"),
        A("email", "a@x"),
        A("email", "b@x"),
    }
    assert sub.nodes[E("A")] == {"type": "ENTITY", "network": "0", "flags": 2}
    assert sub.nodes[E("B")]["flags"] == 0
    assert sub.nodes[A("phone", "111")] == {"type": "phone", "flags": 0}
    assert sub.has_edge(E("A"), A("phone", "111"))
    assert sub.has_edge(E("B"), A("phone", "111"))
    assert A("city", "Metro") not in sub


def test_subgraph_follows_inferred_entity_links(graph):
    links = {E("E"): frozenset({E("F")}), E("F"): frozenset({E("E")})}
    sub = network_subgraph(graph, [E("E")], {E("E"): 3}, None, frozenset(), links)
    assert sub.has_edge(E("E"), E("F"))
    assert sub.nodes[E("F")]["network"] == ""


def test_subgraph_adds_attribute_neighbours_two_hops_out():
    graph = nx.Graph()
    for node in [E("A"), A("owner", "Paris"), A("city", "Paris"), A("city", "Lyon")]:
        graph.add_node(node, type=node.split("==")[0])
    graph.add_edge(E("A"), A("owner", "Paris"))
    graph.add_edge(A("owner", "Paris"), A("city", "Paris"), type="equality")
    links = {A("city", "Paris"): frozenset({A("city", "Lyon")})}
    sub = network_subgraph(graph, [E("A")], links=links)
    # legacy behaviour: the second hop is added as a node, the third hop with an edge
    assert A("city", "Paris") in sub
    assert sub.has_edge(A("city", "Paris"), A("city", "Lyon"))
    assert A("city", "Lyon") in sub


def test_should_merge_on_shared_type_or_value():
    assert should_merge(A("city", "Paris"), A("city", "Lyon"))
    assert should_merge(A("city", "Paris"), A("owner", "Paris"))
    assert not should_merge(A("city", "Paris"), A("owner", "Lyon"))
    assert should_merge(f"{A('a', '1')};{A('b', '2')}", A("b", "9"))


def test_merge_nodes_joins_neighbours_and_keeps_max_flags():
    graph = nx.Graph()
    graph.add_node(A("city", "Paris"), type="city", flags=1)
    graph.add_node(A("owner", "Paris"), type="owner", flags=3)
    graph.add_node(E("A"), type="ENTITY")
    graph.add_edge(A("city", "Paris"), A("owner", "Paris"))
    graph.add_edge(E("A"), A("owner", "Paris"))
    merged = merge_nodes(graph)
    name = f"{A('city', 'Paris')};{A('owner', 'Paris')}"
    assert set(merged.nodes()) == {name, E("A")}
    assert merged.nodes[name] == {"type": "city;owner", "flags": 3}
    assert merged.has_edge(name, E("A"))


def test_simplify_drops_single_degree_attributes(graph, flags):
    sub = network_subgraph(
        graph, [E("A"), E("B")], {}, flags, frozenset({A("city", "Metro")})
    )
    simple = simplify(sub)
    assert set(simple.nodes()) == {E("A"), E("B"), A("phone", "111")}
    assert sub.number_of_nodes() == 5  # input untouched


def test_simplify_merges_then_drops_again(graph):
    sub = network_subgraph(
        graph, [E("C"), E("D")], trimmed=frozenset({A("city", "Metro")})
    )
    simple = simplify(sub)
    # phone==222 and email==c@x both have degree 2 but share neither type nor value
    assert set(simple.nodes()) == {E("C"), E("D"), A("phone", "222"), A("email", "c@x")}


def test_subgraph_frames(graph, flags):
    sub = network_subgraph(
        graph, [E("A")], {E("A"): 0}, flags, frozenset({A("city", "Metro")})
    )
    nodes, edges = subgraph_frames(sub)
    assert nodes.columns == ["node", "type", "flags"]
    assert edges.columns == ["source", "target", "attribute"]
    assert dict(nodes.select("node", "flags").rows())[E("A")] == 2
    assert set(edges.get_column("attribute").to_list()) == {"phone", "email"}
    assert subgraph_frames(nx.Graph())[0].is_empty()
