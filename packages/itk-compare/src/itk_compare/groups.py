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


def melt_attributes(
    frame: pl.DataFrame, id_columns: Sequence[str], aggregates: Sequence[str]
) -> pl.DataFrame:
    """One row per record and non-missing aggregate value, as ``attribute_value``."""
    return (
        _unpivot(frame, list(id_columns), list(aggregates))
        .drop_nulls(subset=["value"])
        .with_columns(
            (pl.col("attribute") + ":" + pl.col("value").cast(pl.Utf8)).alias(
                ATTRIBUTE_VALUE
            )
        )
        .drop(["attribute", "value"])
    )


def _unpivot(frame: pl.DataFrame, index: list[str], on: list[str]) -> pl.DataFrame:
    """``unpivot`` on polars >= 1.0, ``melt`` before it (the lock pins 0.20.10)."""
    unpivot = getattr(frame, "unpivot", None)
    if unpivot is not None:
        return unpivot(
            on=on, index=index, variable_name="attribute", value_name="value"
        )
    return frame.melt(
        id_vars=index, value_vars=on, variable_name="attribute", value_name="value"
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
    counted = (
        melt_attributes(frame, groups, aggregates)
        .group_by(keys)
        .agg(pl.len().cast(pl.Int64).alias(GROUP_ATTRIBUTE_COUNT))
    )
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
