# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import networkx as nx
import pytest
from itk_networks.communities import find_networks, split_component
from itk_networks.models import Network, NetworkConfig
from itk_networks_testing import ChunkDetector, E


def _graph(*edges: tuple[str, str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def test_networks_are_components_largest_first_ties_by_smallest_member():
    graph = _graph(
        (E("Z"), E("Y")), (E("A"), E("B")), (E("M"), E("N")), (E("M"), E("O"))
    )
    networks, membership = find_networks(graph, NetworkConfig())
    assert networks == [
        Network(0, (E("M"), E("N"), E("O"))),
        Network(1, (E("A"), E("B"))),
        Network(2, (E("Y"), E("Z"))),
    ]
    assert membership[E("O")] == 0
    assert membership[E("Z")] == 2


def test_small_components_do_not_need_a_detector():
    graph = _graph((E("A"), E("B")))
    networks, _ = find_networks(graph, NetworkConfig(max_network_entities=2), None)
    assert len(networks) == 1


def test_oversized_component_is_split_with_configured_size_and_seed():
    graph = nx.complete_graph([E(c) for c in "ABCDE"])
    detector = ChunkDetector()
    config = NetworkConfig(max_network_entities=2, leiden_seed=7)
    networks, membership = find_networks(graph, config, detector)
    assert detector.calls == [(5, 2, 7)]
    assert [n.entities for n in networks] == [
        (E("A"), E("B")),
        (E("C"), E("D")),
        (E("E"),),
    ]
    assert membership == {E("A"): 0, E("B"): 0, E("C"): 1, E("D"): 1, E("E"): 2}


def test_split_only_touches_the_oversized_component():
    graph = nx.complete_graph([E(c) for c in "ABC"])
    graph.add_edge(E("X"), E("Y"))
    detector = ChunkDetector()
    networks, _ = find_networks(graph, NetworkConfig(max_network_entities=2), detector)
    assert detector.calls == [(3, 2, 42)]
    assert [n.entities for n in networks] == [
        (E("A"), E("B")),
        (E("C"),),
        (E("X"), E("Y")),
    ]


def test_split_component_rejects_partial_assignment():
    class Forgetful:
        def detect(self, graph, max_size, seed):
            return {E("A"): 0}

    graph = nx.complete_graph([E(c) for c in "ABC"])
    with pytest.raises(ValueError, match="unassigned"):
        split_component(graph, (E("A"), E("B"), E("C")), 2, 0, Forgetful())


def test_detector_sees_nodes_and_edges_in_sorted_order():
    class Recording:
        def detect(self, graph, max_size, seed):
            self.nodes = list(graph.nodes())
            self.edges = list(graph.edges())
            return dict.fromkeys(graph.nodes(), 0)

    graph = nx.Graph()
    graph.add_edges_from([(E("C"), E("A")), (E("B"), E("C")), (E("A"), E("B"))])
    detector = Recording()
    split_component(graph, (E("C"), E("B"), E("A")), 2, 0, detector)
    assert detector.nodes == [E("A"), E("B"), E("C")]
    assert detector.edges == [(E("A"), E("B")), (E("A"), E("C")), (E("B"), E("C"))]


def test_empty_projection_gives_no_networks():
    networks, membership = find_networks(nx.Graph(), NetworkConfig())
    assert networks == []
    assert membership == {}
