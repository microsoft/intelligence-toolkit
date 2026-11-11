# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Converging attribute pairs, their growth into patterns, and pattern scores."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from itk_patterns.embed import is_converging_pair
from itk_patterns.models import (
    ATTRIBUTE_TYPE,
    COUNT,
    DETECTIONS,
    FULL_ATTRIBUTE,
    LENGTH,
    MEAN,
    NODE1,
    NODE2,
    OVERALL_SCORE,
    PAIR_SCHEMA,
    PATTERN,
    PATTERN_SCHEMA,
    PERIOD,
    SUBJECT,
    Z_SCORE,
    join_pattern,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from itk_patterns.models import DetectionConfig, PatternConfig, Positions


class RecordCounter:
    """Counts subjects matching every attribute (and period) in a query.

    Periods and attributes share one namespace of subject-id sets, as in the
    legacy counter; a query is the intersection of the sets it names.
    """

    def __init__(self, links: pl.DataFrame) -> None:
        self.periods: list[str] = sorted(links.get_column(PERIOD).unique().to_list())
        self.attributes: list[str] = sorted(
            links.get_column(FULL_ATTRIBUTE).unique().to_list()
        )
        self._ids: dict[str, frozenset[int]] = {}
        for key_column in (FULL_ATTRIBUTE, PERIOD):
            grouped = links.group_by(key_column).agg(pl.col(SUBJECT).unique())
            for key, ids in grouped.select([key_column, SUBJECT]).rows():
                self._ids[key] = frozenset(ids)
        self._cache: dict[frozenset[str], int] = {}

    def count_records(self, terms: Iterable[str]) -> int:
        key = frozenset(terms)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        ids: frozenset[int] | None = None
        for term in sorted(key):
            matching = self._ids.get(term, frozenset())
            ids = matching if ids is None else ids & matching
            if not ids:
                break
        count = len(ids) if ids else 0
        self._cache[key] = count
        return count

    def period_counts(self, attributes: Sequence[str]) -> list[int]:
        return [self.count_records([p, *attributes]) for p in self.periods]

    def period_mean_sd_max(self, attributes: Sequence[str]) -> tuple[float, float, int]:
        """Population mean, standard deviation and maximum of the per-period counts."""
        counts = self.period_counts(attributes)
        if not counts:
            return 0.0, 0.0, 0
        values = np.asarray(counts, dtype=np.float64)
        return float(values.mean()), float(values.std()), int(values.max())


def node_types(links: pl.DataFrame) -> dict[str, str]:
    return dict(links.select([FULL_ATTRIBUTE, ATTRIBUTE_TYPE]).unique().rows())


def converging_pairs(
    positions: Positions,
    links: pl.DataFrame,
    counter: RecordCounter,
    config: PatternConfig,
    detection: DetectionConfig,
) -> tuple[pl.DataFrame, int]:
    """Pairs of different-type attributes that converge in a period.

    A pair counts when :func:`is_converging_pair` holds and at least
    ``min_pattern_count`` subjects carry both attributes in that period. Also
    returns the number of pairs examined, counting same-type pairs too, as
    the legacy workflow reported.
    """
    nodes = sorted(positions)
    kinds = node_types(links)
    all_pairs = 0
    rows: list[tuple[str, str, str, int, float, float, float | None]] = []
    for period in counter.periods:
        for index, node1 in enumerate(nodes):
            for node2 in nodes[index + 1 :]:
                all_pairs += 1
                if kinds.get(node1) == kinds.get(node2):
                    continue
                if not is_converging_pair(
                    period, node1, node2, positions, config.convergence
                ):
                    continue
                period_count = counter.count_records([period, node1, node2])
                if period_count < detection.min_pattern_count:
                    continue
                mean_count, _, _ = counter.period_mean_sd_max([node1, node2])
                factor = period_count / mean_count if mean_count > 0 else None
                rows.append((
                    period,
                    node1,
                    node2,
                    period_count,
                    mean_count,
                    period_count - mean_count,
                    factor,
                ))
    columns = list(PAIR_SCHEMA)
    frame = pl.DataFrame(
        {name: [row[i] for row in rows] for i, name in enumerate(columns)},
        schema=PAIR_SCHEMA,
    )
    return frame, all_pairs


def grow_patterns(
    pairs: pl.DataFrame, counter: RecordCounter, detection: DetectionConfig
) -> dict[str, list[tuple[tuple[str, ...], int]]]:
    """Extend converging pairs into larger patterns, period by period.

    Starting from the empty pattern, every pair adds the attribute(s) it
    contributes; a candidate survives when every pair within it converges in
    that period, it is no longer than ``max_pattern_length`` and strictly more
    than ``min_pattern_count`` subjects match it in the period. The strict
    comparison is the legacy one. The result maps each period to the patterns
    found, in discovery order, each with its period count.
    """
    found: dict[str, list[tuple[tuple[str, ...], int]]] = {}
    for period in counter.periods:
        period_pairs = [
            (a, b) if a < b else (b, a)
            for a, b in pairs.filter(pl.col(PERIOD) == period)
            .select([NODE1, NODE2])
            .rows()
        ]
        pair_set = set(period_pairs)
        patterns: list[tuple[tuple[str, ...], int]] = []
        seen: set[tuple[str, ...]] = set()
        queue: list[tuple[str, ...]] = [()]
        position = 0
        while position < len(queue):
            pattern = queue[position]
            position += 1
            members = set(pattern)
            for a, b in period_pairs:
                a_in, b_in = a in members, b in members
                if pattern and (a_in == b_in):
                    continue
                if a_in:
                    candidate = tuple(sorted([*pattern, b]))
                elif b_in:
                    candidate = tuple(sorted([*pattern, a]))
                else:
                    candidate = tuple(sorted([*pattern, a, b]))
                if len(candidate) > detection.max_pattern_length or candidate in seen:
                    continue
                if any(pair not in pair_set for pair in combinations(candidate, 2)):
                    continue
                count = counter.count_records([period, *candidate])
                if count > detection.min_pattern_count:
                    seen.add(candidate)
                    queue.append(candidate)
                    patterns.append((candidate, count))
        found[period] = patterns
    return found


def score_patterns(
    found: dict[str, list[tuple[tuple[str, ...], int]]], counter: RecordCounter
) -> pl.DataFrame:
    """The pattern frame: z-scores against the pattern's own period history.

    A pattern whose count never varies across periods has no z-score and is
    dropped, as are negative z-scores. ``overall_score`` is
    ``z_score * length * detections * ln(1 + count)`` divided by its maximum,
    where ``detections`` counts the periods the pattern was found in. Rows are
    sorted by ``overall_score`` descending, then period, then pattern.
    """
    periods: list[str] = []
    names: list[str] = []
    lengths: list[int] = []
    counts: list[int] = []
    means: list[float] = []
    z_scores: list[float] = []
    for period, patterns in found.items():
        for attributes, count in patterns:
            if count <= 0:
                continue
            mean, sd, _ = counter.period_mean_sd_max(attributes)
            if sd <= 0:
                continue
            z = (count - mean) / sd
            if z < 0:
                continue
            periods.append(period)
            names.append(join_pattern(attributes))
            lengths.append(len(attributes))
            counts.append(count)
            means.append(mean)
            z_scores.append(z)
    if not names:
        return pl.DataFrame(schema=PATTERN_SCHEMA)
    frame = pl.DataFrame(
        {
            PERIOD: periods,
            PATTERN: names,
            LENGTH: lengths,
            COUNT: counts,
            MEAN: means,
            Z_SCORE: z_scores,
        },
        schema={
            k: PATTERN_SCHEMA[k]
            for k in (PERIOD, PATTERN, LENGTH, COUNT, MEAN, Z_SCORE)
        },
    )
    frame = frame.with_columns(
        pl.col(PERIOD).count().over(PATTERN).cast(pl.Int64).alias(DETECTIONS)
    )
    raw = (
        pl.col(Z_SCORE)
        * pl.col(LENGTH)
        * pl.col(DETECTIONS)
        * (pl.col(COUNT).cast(pl.Float64) + 1.0).log()
    )
    frame = frame.with_columns(raw.alias(OVERALL_SCORE))
    top = frame.get_column(OVERALL_SCORE).max()
    if top is not None and float(top) > 0:  # pyright: ignore[reportArgumentType]
        frame = frame.with_columns((pl.col(OVERALL_SCORE) / top).alias(OVERALL_SCORE))
    return frame.select([pl.col(c).cast(t) for c, t in PATTERN_SCHEMA.items()]).sort(
        [OVERALL_SCORE, PERIOD, PATTERN], descending=[True, False, False]
    )


def display_patterns(patterns: pl.DataFrame) -> pl.DataFrame:
    """Legacy rounding: ``mean`` to whole numbers, both scores to two places."""
    return patterns.with_columns(
        pl.col(MEAN).round(0),
        pl.col(Z_SCORE).round(2),
        pl.col(OVERALL_SCORE).round(2),
    )


def detect_patterns(
    positions: Positions,
    links: pl.DataFrame,
    config: PatternConfig,
    detection: DetectionConfig,
    counter: RecordCounter | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Converging pairs to scored patterns. Returns ``(patterns, pairs, all_pairs)``."""
    counter = counter or RecordCounter(links)
    pairs, all_pairs = converging_pairs(positions, links, counter, config, detection)
    found = grow_patterns(pairs, counter, detection)
    return score_patterns(found, counter), pairs, all_pairs


__all__ = [
    "RecordCounter",
    "converging_pairs",
    "detect_patterns",
    "display_patterns",
    "grow_patterns",
    "node_types",
    "score_patterns",
]
