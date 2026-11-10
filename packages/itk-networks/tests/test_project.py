# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import networkx as nx
import pytest
from itk_networks.prepare import attribute_links, build_graph
from itk_networks.project import (
    entity_neighbors,
    project_entity_graph,
    trim_attributes,
)
from itk_networks_testing import A, E, attribute_table


@pytest.fixture()
def graph():
    return build_graph(attribute_links(attribute_table()))


def test_trim_attributes_by_degree_and_explicit_removal(graph):
    frame, trimmed = trim_attributes(graph, 3, [A("phone", "111")])
    assert frame.columns == ["Attribute", "Linked Entities"]
    assert frame.rows() == [(A("city", "Metro"), 6)]
    assert trimmed == {A("city", "Metro"), A("phone", "111")}


def test_trim_attributes_sorts_by_degree_then_name(graph):
    frame, _ = trim_attributes(graph, 1)
    assert frame.get_column("Attribute").to_list() == [
        A("email", "c@x"),
        A("phone", "111"),
        A("phone", "222"),
        A("city", "Metro"),
    ]


def test_trim_attributes_never_trims_entities(graph):
    _, trimmed = trim_attributes(graph, 0)
    assert not any(node.startswith("ENTITY") for node in trimmed)


def test_entity_neighbors_merge_graph_and_inferred_links(graph):
    links = {E("A"): frozenset({E("Z")})}
    near = entity_neighbors(graph, links, frozenset({A("city", "Metro")}), E("A"))
    assert near == [E("Z"), A("email", "a@x"), A("phone", "111")]


def test_entity_neighbors_unknown_node_raises(graph):
    with pytest.raises(ValueError, match="not in graph"):
        entity_neighbors(graph, {}, frozenset(), E("nope"))
    assert entity_neighbors(nx.Graph(), {}, frozenset(), E("A")) == []


def test_projection_links_entities_sharing_an_attribute(graph):
    _, trimmed = trim_attributes(graph, 3)
    projected = project_entity_graph(graph, trimmed)
    assert set(projected.edges()) == {(E("A"), E("B")), (E("C"), E("D"))}
    assert E("E") not in projected
    assert E("F") not in projected


def test_projection_without_trimming_connects_everyone_via_city(graph):
    projected = project_entity_graph(graph)
    assert nx.number_connected_components(projected) == 1
    assert projected.number_of_nodes() == 6


def test_projection_respects_supporting_types(graph):
    _, trimmed = trim_attributes(graph, 3)
    projected = project_entity_graph(
        graph, trimmed, supporting_attribute_types=["phone"]
    )
    assert set(projected.edges()) == {(E("C"), E("D"))}


def test_projection_uses_inferred_entity_links(graph):
    _, trimmed = trim_attributes(graph, 3)
    links = {E("E"): frozenset({E("F")}), E("F"): frozenset({E("E")})}
    projected = project_entity_graph(graph, trimmed, links)
    assert projected.has_edge(E("E"), E("F"))


def test_projection_uses_inferred_attribute_links_two_hops(graph):
    _, trimmed = trim_attributes(graph, 3)
    links = {
        A("phone", "333"): frozenset({A("phone", "444")}),
        A("phone", "444"): frozenset({A("phone", "333")}),
    }
    projected = project_entity_graph(graph, trimmed, links)
    assert projected.has_edge(E("E"), E("F"))


def test_projection_uses_equality_edges_three_hops():
    graph = nx.Graph()
    for node, kind in [
        (E("A"), "ENTITY"),
        (E("B"), "ENTITY"),
        (A("owner", "Paris"), "owner"),
        (A("city", "Paris"), "city"),
    ]:
        graph.add_node(node, type=kind)
    graph.add_edge(E("A"), A("owner", "Paris"))
    graph.add_edge(E("B"), A("city", "Paris"))
    graph.add_edge(A("owner", "Paris"), A("city", "Paris"), type="equality")
    projected = project_entity_graph(graph)
    assert projected.has_edge(E("A"), E("B"))
    assert not project_entity_graph(graph, frozenset({A("city", "Paris")})).edges()
