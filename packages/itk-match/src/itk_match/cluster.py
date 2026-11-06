# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Group candidate pairs into entity groups with union-find."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

import polars as pl

from itk_match.models import (
    DATASET,
    GROUP_ID,
    GROUP_SIZE,
    NAME,
    NAME_SIMILARITY,
    Group,
    Pair,
)


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self._parent.setdefault(x, x)
        if parent != x:
            self._parent[x] = self.find(parent)
        return self._parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def components(self) -> list[tuple[int, ...]]:
        members: defaultdict[int, list[int]] = defaultdict(list)
        for x in list(self._parent):
            members[self.find(x)].append(x)
        return [tuple(sorted(rows)) for rows in members.values()]


def cluster(pairs: list[Pair], min_name_similarity: float = 0.0) -> list[Group]:
    """Connected components over pairs whose name similarity meets the threshold.

    Groups are numbered from 0 in descending order of mean name similarity,
    ties broken by their smallest row, so output is deterministic.
    """
    kept = [p for p in pairs if p.name_similarity >= min_name_similarity]
    uf = UnionFind()
    for p in kept:
        uf.union(p.left, p.right)

    scores: defaultdict[int, list[float]] = defaultdict(list)
    for p in kept:
        scores[uf.find(p.left)].append(p.name_similarity)

    def mean_score(rows: tuple[int, ...]) -> float:
        return fmean(scores[uf.find(rows[0])])

    components = sorted(uf.components(), key=lambda r: (-mean_score(r), r[0]))
    return [
        Group(id=i, rows=rows, name_similarity=mean_score(rows))
        for i, rows in enumerate(components)
    ]


def groups_frame(records: pl.DataFrame, groups: list[Group]) -> pl.DataFrame:
    """Matched records with group_id, group_size and name_similarity in front."""
    membership = pl.DataFrame(
        {
            "_row": [row for g in groups for row in g.rows],
            GROUP_ID: [g.id for g in groups for _ in g.rows],
            GROUP_SIZE: [g.size for g in groups for _ in g.rows],
            NAME_SIMILARITY: [g.name_similarity for g in groups for _ in g.rows],
        },
        schema={
            "_row": pl.UInt32,
            GROUP_ID: pl.UInt32,
            GROUP_SIZE: pl.UInt32,
            NAME_SIMILARITY: pl.Float64,
        },
    )
    return (
        membership.join(records.with_row_index("_row"), on="_row", how="inner")
        .drop("_row")
        .sort([GROUP_ID, DATASET, NAME])
    )
