# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Record counts per group and attribute-value counts per group, with ranks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_compare.models import (
    ATTRIBUTE_VALUE,
    GROUP_ATTRIBUTE_COUNT,
    GROUP_ATTRIBUTE_RANK,
    GROUP_COUNT,
    GROUP_RANK,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def group_counts(frame: pl.DataFrame, groups: Sequence[str]) -> pl.DataFrame:
    """One row per group: ``group_count`` and ``group_rank``.

    Rank 1 is the group with most records; tied groups share the lower rank.
    """
    groups = list(groups)
    return (
        frame.group_by(groups)
        .agg(pl.len().cast(pl.Int64).alias(GROUP_COUNT))
        .with_columns(
            pl.col(GROUP_COUNT)
            .rank(method="max", descending=True)
            .cast(pl.Int64)
            .alias(GROUP_RANK)
        )
        .sort(groups)
    )


def count_attribute_values(
    frame: pl.DataFrame,
    id_columns: Sequence[str],
    aggregates: Sequence[str],
    count_name: str,
) -> pl.DataFrame:
    """Count each non-missing ``attribute:value`` per combination of ``id_columns``.

    Counts one aggregate column at a time and builds the ``attribute_value``
    text on the counted rows, not on every record, which is what keeps large
    inputs fast. Returns ``id_columns``, ``attribute_value`` and ``count_name``.
    """
    ids = list(id_columns)
    schema = {
        **{c: frame.schema[c] for c in ids},
        ATTRIBUTE_VALUE: pl.Utf8,
        count_name: pl.Int64,
    }
    if not aggregates:
        return pl.DataFrame(schema=schema)
    lazy = frame.lazy()
    parts = [
        lazy.select([*ids, column])
        .drop_nulls(subset=[column])
        .group_by([*ids, column])
        .agg(pl.len().cast(pl.Int64).alias(count_name))
        .select([
            *ids,
            (pl.lit(f"{column}:") + pl.col(column).cast(pl.Utf8)).alias(
                ATTRIBUTE_VALUE
            ),
            count_name,
        ])
        for column in aggregates
    ]
    # collect_all runs the per-column counts in parallel. Summing again merges
    # the rare case of two columns rendering the same ``attribute:value`` text
    # (a column name that itself contains a colon).
    return (
        pl.concat(pl.collect_all(parts))
        .group_by([*ids, ATTRIBUTE_VALUE])
        .agg(pl.col(count_name).sum().cast(pl.Int64))
    )


def attribute_counts(
    frame: pl.DataFrame, groups: Sequence[str], aggregates: Sequence[str]
) -> pl.DataFrame:
    """Every group crossed with every attribute value seen in the data.

    ``group_attribute_count`` is 0 where a group never shows the value;
    ``group_attribute_rank`` ranks groups per attribute value (1 = highest
    count; ties share the lower rank).
    """
    groups = list(groups)
    keys = [*groups, ATTRIBUTE_VALUE]
    counted = count_attribute_values(frame, groups, aggregates, GROUP_ATTRIBUTE_COUNT)
    complete = (
        frame.select(groups)
        .unique()
        .join(counted.select(ATTRIBUTE_VALUE).unique(), how="cross")
        .join(counted, on=keys, how="left")
        .with_columns(pl.col(GROUP_ATTRIBUTE_COUNT).fill_null(0).cast(pl.Int64))
    )
    return complete.with_columns(
        pl.col(GROUP_ATTRIBUTE_COUNT)
        .rank(method="max", descending=True)
        .over(ATTRIBUTE_VALUE)
        .cast(pl.Int64)
        .alias(GROUP_ATTRIBUTE_RANK)
    ).sort(keys)
