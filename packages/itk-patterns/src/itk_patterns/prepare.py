# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Turn a case table into the long frame of ``(subject, period, attribute)`` links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_patterns.models import (
    ATTRIBUTE_TYPE,
    ATTRIBUTE_VALUE,
    FULL_ATTRIBUTE,
    LINK_SCHEMA,
    PERIOD,
    SUBJECT,
    SUBJECT_ID_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itk_patterns.models import PatternConfig

EMPTY = ""


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


def attribute_columns(frame: pl.DataFrame, config: PatternConfig) -> list[str]:
    """The columns that become attribute nodes, in frame order."""
    if config.period_column not in frame.columns:
        msg = f"Frame has no period column {config.period_column!r}"
        raise ValueError(msg)
    if config.attribute_columns is None:
        return [
            c
            for c in frame.columns
            if c not in (SUBJECT_ID_COLUMN, config.period_column)
        ]
    missing = sorted(set(config.attribute_columns) - set(frame.columns))
    if missing:
        msg = f"Frame has no attribute columns {missing}"
        raise ValueError(msg)
    if config.period_column in config.attribute_columns:
        msg = f"Period column {config.period_column!r} cannot also be an attribute"
        raise ValueError(msg)
    return [c for c in frame.columns if c in config.attribute_columns]


def _unpivot(
    frame: pl.DataFrame,
    index: Sequence[str],
    on: Sequence[str],
    variable_name: str,
    value_name: str,
) -> pl.DataFrame:
    # polars 1.x renamed melt to unpivot and dropped melt in 2.x.
    unpivot = getattr(frame, "unpivot", None)
    if unpivot is not None:
        return unpivot(
            on=list(on),
            index=list(index),
            variable_name=variable_name,
            value_name=value_name,
        )
    return frame.melt(
        id_vars=list(index),
        value_vars=list(on),
        variable_name=variable_name,
        value_name=value_name,
    )


def attribute_links(frame: pl.DataFrame, config: PatternConfig) -> pl.DataFrame:
    """One row per ``(subject, period, attribute)`` with a non-empty value.

    Expects :func:`prepare_frame` output. Subjects are numbered from 1 in row
    order, as the legacy model did; rows with an empty period are dropped and
    so are empty attribute values, which the legacy model kept as ``"nan"``
    nodes when pandas read them as missing.
    """
    columns = attribute_columns(frame, config)
    if not columns:
        msg = "No attribute columns to model"
        raise ValueError(msg)
    base = frame.with_row_index(SUBJECT, offset=1).select(
        pl.col(SUBJECT).cast(pl.Int64),
        pl.col(config.period_column).cast(pl.Utf8).alias(PERIOD),
        *(pl.col(c).cast(pl.Utf8) for c in columns),
    )
    base = base.filter(pl.col(PERIOD).is_not_null() & (pl.col(PERIOD) != EMPTY))
    long = _unpivot(base, [SUBJECT, PERIOD], columns, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE)
    long = long.filter(
        pl.col(ATTRIBUTE_VALUE).is_not_null() & (pl.col(ATTRIBUTE_VALUE) != EMPTY)
    )
    return (
        long.with_columns(
            (
                pl.col(ATTRIBUTE_TYPE)
                + pl.lit(config.type_value_separator)
                + pl.col(ATTRIBUTE_VALUE)
            ).alias(FULL_ATTRIBUTE)
        )
        .select([pl.col(c).cast(t) for c, t in LINK_SCHEMA.items()])
        .sort([SUBJECT, PERIOD, ATTRIBUTE_TYPE])
    )


def node_labels(links: pl.DataFrame) -> dict[str, int]:
    """Each attribute node mapped to the code of its type.

    Codes number the sorted attribute types from 0, which is what the graph
    encoder embedding uses as node classes.
    """
    pairs = (
        links.select([FULL_ATTRIBUTE, ATTRIBUTE_TYPE])
        .unique()
        .sort(FULL_ATTRIBUTE)
        .rows()
    )
    types = sorted({kind for _, kind in pairs})
    code = {kind: index for index, kind in enumerate(types)}
    return {node: code[kind] for node, kind in pairs}


__all__ = ["attribute_columns", "attribute_links", "node_labels", "prepare_frame"]
