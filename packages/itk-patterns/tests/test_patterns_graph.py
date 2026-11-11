# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import math

import pytest
from itk_patterns.graph import (
    build_graph,
    edge_frame,
    period_attribute_sets,
    period_graphs,
)
from itk_patterns.models import EDGE_SCHEMA, SOURCE, TARGET, WEIGHT


def test_edge_frame_scales_observed_pairs_and_fills_missing_ones():
    sets = [["a", "b"], ["a", "b"], ["b", "c"]]
    edges = edge_frame(sets, ["a", "b", "c", "d"], 0.1, 0.5)
    assert edges.schema == EDGE_SCHEMA
    weights = {(s, t): w for s, t, w in edges.select([SOURCE, TARGET, WEIGHT]).rows()}
    # count 2 -> 1.0, count 1 -> min_edge_weight, unobserved -> 0.5 * 0.1
    assert weights["a", "b"] == pytest.approx(1.0)
    assert weights["b", "c"] == pytest.approx(0.1)
    assert weights["a", "c"] == pytest.approx(0.05)
    assert weights["a", "d"] == pytest.approx(0.05)
    assert len(weights) == math.comb(4, 2)


def test_edge_frame_sorted_by_weight_then_endpoints():
    edges = edge_frame([["b", "c"], ["a", "b"]], ["a", "b", "c"], 0.001, 0.1)
    assert edges.select([SOURCE, TARGET]).rows() == [("a", "b"), ("b", "c"), ("a", "c")]


def test_edge_frame_equal_counts_all_weigh_one():
    edges = edge_frame([["a", "b"], ["c", "d"]], ["a", "b", "c", "d"], 0.001, 0.1)
    observed = edges.filter(edges.get_column(WEIGHT) > 0.5)
    assert observed.get_column(WEIGHT).to_list() == [1.0, 1.0]


def test_edge_frame_without_observed_pairs_is_all_missing():
    edges = edge_frame([["a"], ["b"]], ["a", "b"], 0.001, 0.1)
    assert edges.rows() == [("a", "b", pytest.approx(0.0001))]


def test_edge_frame_ignores_duplicate_attributes_within_a_set():
    edges = edge_frame([["a", "a", "b"]], ["a", "b"], 0.001, 0.1)
    assert edges.height == 1


def test_build_graph_includes_isolated_nodes_and_weights():
    edges = edge_frame([["a", "b"]], ["a", "b"], 0.001, 0.1)
    graph = build_graph(edges, ["a", "b", "z"])
    assert set(graph.nodes) == {"a", "b", "z"}
    assert graph.edges["a", "b"]["weight"] == pytest.approx(1.0)


def test_period_attribute_sets_group_by_subject(prepared):
    sets = period_attribute_sets(prepared.links, "P1")
    assert sets[0] == ["city=A", "issue=low", "product=X"]
    assert len(sets) == 4


def test_period_graphs_span_all_attributes_every_period(prepared, config):
    graphs = period_graphs(prepared.links, config)
    assert list(graphs) == ["P1", "P2", "P3"]
    for graph in graphs.values():
        assert sorted(graph.nodes) == prepared.attributes
        assert graph.number_of_edges() == math.comb(len(prepared.attributes), 2)
    # city=A & product=X co-occur most in P3 so carry the top weight there
    assert graphs["P3"].edges["city=A", "product=X"]["weight"] == pytest.approx(1.0)
