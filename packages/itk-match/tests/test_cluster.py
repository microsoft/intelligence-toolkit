# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pytest
from itk_match.cluster import UnionFind, cluster, groups_frame
from itk_match.models import Pair
from itk_match.unify import unify


def test_union_find_merges_transitively():
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(7, 8)
    assert sorted(uf.components()) == [(1, 2, 3), (7, 8)]


def test_cluster_builds_connected_components_with_mean_similarity():
    pairs = [
        Pair(0, 1, 0.01, 1.0),
        Pair(1, 2, 0.02, 0.5),
        Pair(5, 6, 0.01, 0.9),
    ]
    groups = cluster(pairs)

    assert [(g.id, g.rows) for g in groups] == [(0, (5, 6)), (1, (0, 1, 2))]
    assert groups[0].name_similarity == pytest.approx(0.9)
    assert groups[1].name_similarity == pytest.approx(0.75)


def test_cluster_drops_pairs_below_name_threshold():
    pairs = [Pair(0, 1, 0.01, 1.0), Pair(1, 2, 0.02, 0.5)]
    groups = cluster(pairs, min_name_similarity=0.75)
    assert [g.rows for g in groups] == [(0, 1)]


def test_cluster_of_nothing_is_empty():
    assert cluster([]) == []
    assert cluster([Pair(0, 1, 0.0, 0.1)], min_name_similarity=0.5) == []


def test_cluster_is_deterministic_regardless_of_pair_order():
    pairs = [Pair(0, 1, 0.01, 0.8), Pair(2, 3, 0.01, 0.8), Pair(1, 4, 0.01, 0.8)]
    assert cluster(pairs) == cluster(list(reversed(pairs)))


def test_groups_frame_puts_group_columns_first(dataset_one, dataset_two):
    records = unify([dataset_one, dataset_two])
    groups = cluster([Pair(0, 3, 0.0, 1.0), Pair(1, 4, 0.0, 0.5)])
    frame = groups_frame(records, groups)

    assert frame.columns[:3] == ["group_id", "group_size", "name_similarity"]
    assert frame.height == 4
    first = frame.filter(frame["group_id"] == 0)
    assert first.get_column("record_id").to_list() == ["D1:1", "D2:0"]
    assert first.get_column("group_size").to_list() == [2, 2]


def test_groups_frame_with_no_groups_is_empty_but_typed(dataset_one):
    frame = groups_frame(unify([dataset_one]), [])
    assert frame.height == 0
    assert "group_id" in frame.columns
