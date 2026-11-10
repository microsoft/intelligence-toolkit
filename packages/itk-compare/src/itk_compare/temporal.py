# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Attribute-value counts per temporal window, their deltas and ranks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_compare.groups import melt_attributes
from itk_compare.models import ATTRIBUTE_VALUE, window_columns

if TYPE_CHECKING:
    from collections.abc import Sequence


def window_counts(
    frame: pl.DataFrame,
    groups: Sequence[str],
    temporal: str,
    aggregates: Sequence[str],
) -> pl.DataFrame:
    """Count each attribute value per group and window.

    Records without a window value are skipped.
    """
    columns = window_columns(temporal)
    keys = [*groups, columns.window, ATTRIBUTE_VALUE]
    return (
        melt_attributes(frame, [*groups, temporal], aggregates)
        .drop_nulls(subset=[temporal])
        .rename({temporal: columns.window})
        .group_by(keys)
        .agg(pl.len().cast(pl.Int64).alias(columns.count))
        .sort(keys)
    )


def complete_windows(
    counts: pl.DataFrame,
    groups: Sequence[str],
    temporal: str,
    group_keys: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Fill every group x window x attribute value with a 0 count where absent.

    Groups come from ``group_keys`` (any frame holding the group columns)
    when given, else from the counts. Attribute values a group never shows
    in any window are dropped again: they carry no change to report.
    """
    groups = list(groups)
    columns = window_columns(temporal)
    keys = [*groups, columns.window, ATTRIBUTE_VALUE]
    if counts.height == 0:
        return counts
    group_frame = (
        group_keys.select(groups).unique()
        if group_keys is not None
        else counts.select(groups).unique()
    )
    complete = (
        group_frame.join(counts.select(columns.window).unique(), how="cross")
        .join(counts.select(ATTRIBUTE_VALUE).unique(), how="cross")
        .join(counts, on=keys, how="left")
        .with_columns(pl.col(columns.count).fill_null(0).cast(pl.Int64))
    )
    totals = complete.group_by([*groups, ATTRIBUTE_VALUE]).agg(
        pl.col(columns.count).sum().alias("_total")
    )
    present = totals.filter(pl.col("_total") > 0).drop("_total")
    return complete.join(present, on=[*groups, ATTRIBUTE_VALUE], how="inner").sort(keys)


def window_deltas(
    complete: pl.DataFrame, groups: Sequence[str], temporal: str
) -> pl.DataFrame:
    """Change in count from the previous window of the same group and attribute value.

    The first window of each series gets 0.
    """
    columns = window_columns(temporal)
    return complete.sort([*groups, ATTRIBUTE_VALUE, columns.window]).with_columns(
        pl.col(columns.count)
        .diff()
        .over([*groups, ATTRIBUTE_VALUE])
        .fill_null(0)
        .cast(pl.Int64)
        .alias(columns.delta)
    )


def window_ranks(complete: pl.DataFrame, temporal: str) -> pl.DataFrame:
    """Dense rank of groups per window and attribute value (1 = highest count)."""
    columns = window_columns(temporal)
    return complete.with_columns(
        pl.col(columns.count)
        .rank(method="dense", descending=True)
        .over([columns.window, ATTRIBUTE_VALUE])
        .cast(pl.Int64)
        .alias(columns.rank)
    )


def temporal_frame(
    frame: pl.DataFrame,
    groups: Sequence[str],
    temporal: str,
    aggregates: Sequence[str],
) -> pl.DataFrame:
    """Counts, ranks and deltas per group, window and attribute value."""
    groups = list(groups)
    columns = window_columns(temporal)
    counts = window_counts(frame, groups, temporal, aggregates)
    complete = complete_windows(counts, groups, temporal, group_keys=frame)
    if complete.height == 0:
        return pl.DataFrame(
            schema={
                **dict.fromkeys(groups, pl.Utf8),
                columns.window: pl.Utf8,
                ATTRIBUTE_VALUE: pl.Utf8,
                columns.count: pl.Int64,
                columns.rank: pl.Int64,
                columns.delta: pl.Int64,
            }
        )
    ranked = window_ranks(window_deltas(complete, groups, temporal), temporal)
    return ranked.select([
        *groups,
        columns.window,
        ATTRIBUTE_VALUE,
        columns.count,
        columns.rank,
        columns.delta,
    ]).sort([*groups, ATTRIBUTE_VALUE, columns.window])
