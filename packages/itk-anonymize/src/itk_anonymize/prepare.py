# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Turn an arbitrary frame into the categorical text table a DP engine expects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_anonymize.models import RawTable

if TYPE_CHECKING:
    from collections.abc import Sequence

EMPTY = ""
_PRESENCE_ONLY = ["0", "0.0", "False"]


def prepare_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a copy with every column as Utf8 and nulls as ``""``.

    Float columns whose non-null values are all whole numbers are written as
    integers (``3.0`` becomes ``"3"``), as the legacy ``fix_null_ints`` did;
    NaN counts as null. Booleans become ``"True"``/``"False"`` to match pandas.
    """
    expressions = []
    for name, dtype in frame.schema.items():
        column = pl.col(name)
        if dtype in (pl.Float32, pl.Float64):
            column = column.fill_nan(None)
            finite = frame.get_column(name).fill_nan(None).drop_nulls()
            whole = finite.len() > 0 and bool((finite == finite.floor()).all())
            text = (
                column.cast(pl.Int64).cast(pl.Utf8) if whole else column.cast(pl.Utf8)
            )
        elif dtype == pl.Boolean:
            text = (
                pl.when(column)
                .then(pl.lit("True"))
                .when(column.is_not_null())
                .then(pl.lit("False"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
            )
        else:
            text = column.cast(pl.Utf8)
        expressions.append(text.fill_null(EMPTY).alias(name))
    return frame.select(expressions)


def suppress_boolean_binary(frame: pl.DataFrame) -> pl.DataFrame:
    """Blank ``0``/``False`` in columns that only carry presence information.

    A column qualifies when it has at most two distinct non-empty values and
    one of them is ``0``, ``0.0`` or ``False``. Expects :func:`prepare_frame`
    output.
    """
    expressions = []
    for name in frame.columns:
        distinct = set(frame.get_column(name).unique().to_list()) - {EMPTY, None}
        if len(distinct) <= 2 and distinct & set(_PRESENCE_ONLY):
            expressions.append(
                pl.when(pl.col(name).is_in(_PRESENCE_ONLY))
                .then(pl.lit(EMPTY))
                .otherwise(pl.col(name))
                .alias(name)
            )
        else:
            expressions.append(pl.col(name))
    return frame.select(expressions)


def suppress_rare_values(
    frame: pl.DataFrame, min_count: int, columns: Sequence[str] | None = None
) -> pl.DataFrame:
    """Blank values that occur fewer than ``min_count`` times in their column.

    ``min_count <= 1`` leaves the frame unchanged, as the legacy UI's ``0`` did.
    Expects :func:`prepare_frame` output.
    """
    if min_count <= 1:
        return frame
    names = set(frame.columns if columns is None else columns)
    expressions = []
    for name in frame.columns:
        if name not in names:
            expressions.append(pl.col(name))
            continue
        counts = frame.get_column(name).value_counts()
        rare = (
            counts.filter((pl.col("count") < min_count) & (pl.col(name) != EMPTY))
            .get_column(name)
            .to_list()
        )
        expressions.append(
            pl.when(pl.col(name).is_in(rare))
            .then(pl.lit(EMPTY))
            .otherwise(pl.col(name))
            .alias(name)
        )
    return frame.select(expressions)


def to_raw_table(frame: pl.DataFrame) -> RawTable:
    """Prepared frame as header plus rows of text."""
    return RawTable.from_frame(frame)
