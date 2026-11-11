# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Frames the pattern charts and reports are drawn from."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_patterns.detect import RecordCounter
from itk_patterns.models import (
    ATTRIBUTE_COUNT_SCHEMA,
    ATTRIBUTE_VALUE,
    COUNT,
    PATTERN,
    PERIOD,
    SUBJECT_ID_COLUMN,
    TIME_SERIES_SCHEMA,
    split_pattern,
)
from itk_patterns.prepare import EMPTY, attribute_columns

if TYPE_CHECKING:
    from collections.abc import Iterable

    from itk_patterns.models import PatternConfig


def time_series(
    links: pl.DataFrame, patterns: Iterable[str], counter: RecordCounter | None = None
) -> pl.DataFrame:
    """Per-period subject counts of each distinct pattern, sorted by pattern, period."""
    counter = counter or RecordCounter(links)
    names = sorted(set(patterns))
    periods: list[str] = []
    labels: list[str] = []
    counts: list[int] = []
    for name in names:
        attributes = split_pattern(name)
        for period, count in zip(
            counter.periods, counter.period_counts(attributes), strict=True
        ):
            periods.append(period)
            labels.append(name)
            counts.append(count)
    return pl.DataFrame(
        {PERIOD: periods, PATTERN: labels, COUNT: counts}, schema=TIME_SERIES_SCHEMA
    )


def pattern_filter(
    pattern: str, period: str, config: PatternConfig, columns: Iterable[str]
) -> pl.Expr:
    """Rows of a prepared frame in ``period`` that carry every attribute of ``pattern``."""
    known = set(columns)
    expression = pl.col(config.period_column) == period
    for attribute in split_pattern(pattern):
        name, separator, value = attribute.partition(config.type_value_separator)
        if not separator or name not in known:
            continue
        expression = expression & (pl.col(name) == value)
    return expression


def attribute_counts(
    source: pl.DataFrame, pattern: str, period: str, config: PatternConfig
) -> pl.DataFrame:
    """How many matching records carry each attribute value, for one pattern and period.

    ``source`` is the :func:`itk_patterns.prepare.prepare_frame` output. Every
    attribute column counts, including the pattern's own, whose count equals
    the number of matching records. Sorted by count descending, then value.
    """
    columns = attribute_columns(source, config)
    matching = source.filter(pattern_filter(pattern, period, config, source.columns))
    if matching.height == 0 or not columns:
        return pl.DataFrame(schema=ATTRIBUTE_COUNT_SCHEMA)
    parts = [
        matching.select(pl.col(column))
        .filter(pl.col(column) != EMPTY)
        .group_by(column)
        .agg(pl.len().cast(pl.Int64).alias(COUNT))
        .select(
            (pl.lit(f"{column}{config.type_value_separator}") + pl.col(column)).alias(
                ATTRIBUTE_VALUE
            ),
            COUNT,
        )
        for column in columns
        if column != SUBJECT_ID_COLUMN
    ]
    return (
        pl.concat(parts)
        .select([pl.col(c).cast(t) for c, t in ATTRIBUTE_COUNT_SCHEMA.items()])
        .sort([COUNT, ATTRIBUTE_VALUE], descending=[True, False])
    )


__all__ = ["attribute_counts", "pattern_filter", "time_series"]
