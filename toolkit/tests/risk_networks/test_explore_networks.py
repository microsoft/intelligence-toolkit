# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl
import pytest
from networkx import Graph

from toolkit.risk_networks.explore_networks import (
    _build_fuzzy_neighbors,
    _integrate_flags,
    _merge_condition,
    _merge_node_list,
    _merge_nodes,
    get_entity_graph,
    get_type_color,
    hsl_to_hex,
    simplify_entities_graph,
)


@pytest.fixture()
def graph() -> Graph:
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


@pytest.fixture()
def simple_graph() -> Graph:
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


class TestFuzzyNeighbors:
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
        result = _build_fuzzy_neighbors(
            nx.Graph(), nx.Graph(), "ENTITY==1", set(), dict
        )
        assert result.nodes == nx.Graph().nodes

    def test_node_inexistent(self, simple_graph):
        with pytest.raises(ValueError, match="Node ENTITY==78 not in graph"):
            _build_fuzzy_neighbors(simple_graph, nx.Graph(), "ENTITY==78", set(), dict)

    def test_fuzzy_neighbor_basic(self, simple_graph):
        network_graph = nx.Graph()
        att_neighbor = "Attr==Type1"
        trimmed_nodeset = set()
        inferred_links = {}
        result = _build_fuzzy_neighbors(
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
        result = _build_fuzzy_neighbors(
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
        result = _build_fuzzy_neighbors(
            simple_graph, network_graph, att_neighbor, trimmed_nodeset, inferred_links
        )

        assert len(result.nodes()) == 0
        assert len(result.edges()) == 0

    def test_fuzzy_neighbor_graph_existent(self, simple_graph, existing_network_graph):
        att_neighbor = "Attr==Type1"
        trimmed_nodeset = set()
        inferred_links = {}
        result = _build_fuzzy_neighbors(
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


class TestIntegrateFlags:
    @pytest.fixture()
    def graph_flags(self):
        G = nx.Graph()
        G.add_node("A")

        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_node("E")
        G.add_node("F")
        return G

    def test_empty_graph(self):
        result = _integrate_flags(nx.Graph(), pl.DataFrame())

        assert len(result.nodes()) == 0

    def test_empty_flags(self, graph_flags):
        result = _integrate_flags(graph_flags, pl.DataFrame())

        assert len(result.nodes()) == 0

    def test_integration(self, graph_flags):
        flags = pl.DataFrame(
            {
                "qualified_entity": ["A", "C", "D", "F"],
                "count": [1, 2, 3, 0],
            }
        )

        result = _integrate_flags(graph_flags, flags)

        assert result.nodes["A"]["flags"] == 1
        assert result.nodes["C"]["flags"] == 2
        assert result.nodes["D"]["flags"] == 3
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]

    def test_sum(self, graph_flags):
        flags = pl.DataFrame(
            {
                "qualified_entity": ["A", "C", "D", "F", "A"],
                "count": [1, 2, 3, 0, 5],
            }
        )

        result = _integrate_flags(graph_flags, flags)

        assert result.nodes["A"]["flags"] == 6
        assert result.nodes["C"]["flags"] == 2
        assert result.nodes["D"]["flags"] == 3
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]

    def test_node_not_in_graph(self, graph_flags):
        flags = pl.DataFrame(
            {
                "qualified_entity": ["A", "C", "D", "F", "Z"],
                "count": [1, 2, 3, 0, 3],
            }
        )

        result = _integrate_flags(graph_flags, flags)

        assert result.nodes["A"]["flags"] == 1
        assert result.nodes["C"]["flags"] == 2
        assert result.nodes["D"]["flags"] == 3
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]
        assert "Z" not in result.nodes()
        assert "flags" not in result.nodes["B"]
        assert "flags" not in result.nodes["E"]
        assert "flags" not in result.nodes["F"]
        assert "Z" not in result.nodes()


class TestMergeCondition:
    def test_merge(self) -> None:
        x = "A==1;B==2;C==3"
        y = "D==4;E==1;F==5"
        assert _merge_condition(x, y) is True

    def test_merge_common_attr(self) -> None:
        x = "A==1;B==2;C==3"
        y = "D==4;E==5;F==2"
        assert _merge_condition(x, y) is True

    def test_merge_not_common(self) -> None:
        x = "A==1;B==2;C==3"
        y = "D==4;E==5;F==6"
        assert _merge_condition(x, y) is False

    def test_merge_multiple_common(self) -> None:
        x = "A==1;B==2;C==3"
        y = "D==4;E==1;F==2"
        assert _merge_condition(x, y) is True


class TestMergeNodeList:
    def test_merge_node_list(self, graph):
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


class TestMergeNodes:
    def test_merge_all_nodes(self, graph):
        merged_graph = _merge_nodes(graph, lambda _x, _y: True)
        assert len(merged_graph.nodes()) < len(graph.nodes())

    def test_merge_no_nodes(self, graph):
        merged_graph = _merge_nodes(graph, lambda _x, _y: False)
        assert len(merged_graph.nodes()) == len(graph.nodes())

    def test_empty_graph(self):
        G = nx.Graph()
        merged_graph = _merge_nodes(G, lambda _x, _y: True)
        assert len(merged_graph.nodes()) == 0


class TestSimplifyEntitiesGraph:
    def test_simplify_condition_false(self, mocker, graph):
        mocker.patch(
            "toolkit.risk_networks.explore_networks._merge_nodes"
        ).return_value = graph

        aba = simplify_entities_graph(graph)
        assert len(aba.nodes()) == 3

    def test_simplify_condition_true(self, mocker, graph):
        G = nx.Graph()
        mocker.patch(
            "toolkit.risk_networks.explore_networks._merge_nodes"
        ).return_value = G

        aba = simplify_entities_graph(graph)
        assert len(aba.nodes()) == 0
        assert len(aba.nodes()) == 0


class TestHslToHex:
    def test_colors(self):
        assert hsl_to_hex(0, 0, 0) == "#000000"
        assert hsl_to_hex(0, 0, 100) == "#ffffff"
        assert hsl_to_hex(0, 100, 50) == "#ff0000"
        assert hsl_to_hex(120, 100, 50) == "#00ff00"
        assert hsl_to_hex(240, 100, 50) == "#0000ff"


class TestGetTypeColor:
    @pytest.fixture()
    def attribute_types(self) -> list[str]:
        return ["Type1", "Type2", "Type3"]

    def test_not_flagged(self, attribute_types) -> None:
        node_type = "Type1"
        is_flagged = False
        result = get_type_color(node_type, is_flagged, attribute_types)

        assert result == "#a8b4ef"

    def test_flagged(self, attribute_types) -> None:
        node_type = "Type1"
        is_flagged = True
        result = get_type_color(node_type, is_flagged, attribute_types)

        assert result == "#efa8a8"

    def test_type_2_3_types(self, attribute_types) -> None:
        node_type = "Type2"
        is_flagged = False
        result = get_type_color(node_type, is_flagged, attribute_types)

        assert result == "#efd3a8"

    def test_type_2_2_types(self, attribute_types) -> None:
        node_type = "Type2"
        is_flagged = False
        attribute_types = attribute_types[:-1]
        result = get_type_color(node_type, is_flagged, attribute_types)

        assert result == "#d1efa8"


class TestGetEntityGraph:
    @pytest.fixture()
    def simple_graph(self) -> Graph:
        G = nx.Graph()
        G.add_node("ENTITY==1")
        G.add_node("ENTITY==2")
        G.add_node("ENTITY==3")
        G.add_node("Attr==Type1")
        G.add_node("Attr==Type2")
        G.add_node("AttributeABCD==Type35")
        G.add_node("AttributeABCD==Type37")

        G.add_edge("ENTITY==1", "ENTITY==2")
        G.add_edge("Attr==Type1", "ENTITY==1")
        G.add_edge("AttributeABCD==Type37", "Attr==Type1")
        G.add_edge("ENTITY==3", "Attr==Type1")
        G.add_edge("ENTITY==3", "AttributeABCD==Type35")
        return G

    def test_empty(self) -> None:
        G = nx.Graph()
        selected = ""
        attribute_types = []

        nodes, edges = get_entity_graph(G, selected, attribute_types)
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_none_selected_nodes(self, simple_graph) -> None:
        attribute_types = ["ENTITY", "AttributeABCD", "Attr"]
        selected = ""
        nodes, _ = get_entity_graph(simple_graph, selected, attribute_types)

        expected_nodes = [
            {
                "title": "ENTITY==1\nFlags: 0",
                "id": "ENTITY==1",
                "label": "1\n(ENTITY)",
                "size": 12,
                "color": "#a8b4ef",
                "font": {"vadjust": -22, "size": 5},
            },
            {
                "title": "ENTITY==3\nFlags: 0",
                "id": "ENTITY==3",
                "label": "3\n(ENTITY)",
                "size": 12,
                "color": "#a8b4ef",
                "font": {"vadjust": -22, "size": 5},
            },
            {
                "title": "AttributeABCD==Type37\nFlags: 0",
                "id": "AttributeABCD==Type37",
                "label": "Type37\n(AttributeABCD)",
                "size": 8,
                "color": "#efd3a8",
                "font": {"vadjust": -18, "size": 5},
            },
            {
                "title": "ENTITY==2\nFlags: 0",
                "id": "ENTITY==2",
                "label": "2\n(ENTITY)",
                "size": 12,
                "color": "#a8b4ef",
                "font": {"vadjust": -22, "size": 5},
            },
            {
                "title": "Attr==Type1\nFlags: 0",
                "id": "Attr==Type1",
                "label": "Type1\n(Attr)",
                "size": 8,
                "color": "#ebefa8",
                "font": {"vadjust": -18, "size": 5},
            },
            {
                "title": "AttributeABCD==Type35\nFlags: 0",
                "id": "AttributeABCD==Type35",
                "label": "Type35\n(AttributeABCD)",
                "size": 8,
                "color": "#efd3a8",
                "font": {"vadjust": -18, "size": 5},
            },
        ]

        expected_nodes = sorted(expected_nodes, key=lambda x: x["id"])
        nodes = sorted(nodes, key=lambda x: x["id"])
        assert expected_nodes == nodes

    def test_none_selected_edges(self, simple_graph):
        attribute_types = ["ENTITY", "AttributeABCD", "Attr"]
        selected = ""
        _, edges = get_entity_graph(simple_graph, selected, attribute_types)

        expected_edges = [
            {
                "source": "Attr==Type1",
                "target": "AttributeABCD==Type37",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==1",
                "target": "Attr==Type1",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==1",
                "target": "ENTITY==2",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==3",
                "target": "Attr==Type1",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==3",
                "target": "AttributeABCD==Type35",
                "color": "mediumgray",
                "size": 1,
            },
        ]

        edges = sorted(edges, key=lambda x: (x["source"], x["target"]))
        assert edges == expected_edges

    def test_selected_nodes(self, simple_graph):
        attribute_types = ["ENTITY", "AttributeABCD", "Attr"]
        selected = "ENTITY==3"
        nodes, _ = get_entity_graph(simple_graph, selected, attribute_types)

        expected_nodes = [
            {
                "title": "ENTITY==1\nFlags: 0",
                "id": "ENTITY==1",
                "label": "1\n(ENTITY)",
                "size": 12,
                "color": "#a8b4ef",
                "font": {"vadjust": -22, "size": 5},
            },
            {
                "title": "ENTITY==3\nFlags: 0",
                "id": "ENTITY==3",
                "label": "3\n(ENTITY)",
                "size": 20,
                "color": "#a8b4ef",
                "font": {"vadjust": -30, "size": 5},
            },
            {
                "title": "AttributeABCD==Type37\nFlags: 0",
                "id": "AttributeABCD==Type37",
                "label": "Type37\n(AttributeABCD)",
                "size": 8,
                "color": "#efd3a8",
                "font": {"vadjust": -18, "size": 5},
            },
            {
                "title": "ENTITY==2\nFlags: 0",
                "id": "ENTITY==2",
                "label": "2\n(ENTITY)",
                "size": 12,
                "color": "#a8b4ef",
                "font": {"vadjust": -22, "size": 5},
            },
            {
                "title": "Attr==Type1\nFlags: 0",
                "id": "Attr==Type1",
                "label": "Type1\n(Attr)",
                "size": 8,
                "color": "#ebefa8",
                "font": {"vadjust": -18, "size": 5},
            },
            {
                "title": "AttributeABCD==Type35\nFlags: 0",
                "id": "AttributeABCD==Type35",
                "label": "Type35\n(AttributeABCD)",
                "size": 8,
                "color": "#efd3a8",
                "font": {"vadjust": -18, "size": 5},
            },
        ]

        nodes = sorted(nodes, key=lambda x: x["id"])
        expected_nodes = sorted(expected_nodes, key=lambda x: x["id"])
        assert nodes == expected_nodes

    def test_selected_edges(self, simple_graph):
        attribute_types = ["ENTITY", "AttributeABCD", "Attr"]
        selected = "ENTITY==3"
        _, edges = get_entity_graph(simple_graph, selected, attribute_types)

        expected_edges = [
            {
                "source": "Attr==Type1",
                "target": "AttributeABCD==Type37",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==1",
                "target": "Attr==Type1",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==1",
                "target": "ENTITY==2",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==3",
                "target": "Attr==Type1",
                "color": "mediumgray",
                "size": 1,
            },
            {
                "source": "ENTITY==3",
                "target": "AttributeABCD==Type35",
                "color": "mediumgray",
                "size": 1,
            },
        ]

        edges = sorted(edges, key=lambda x: (x["source"], x["target"]))
        assert edges == expected_edges
