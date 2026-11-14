# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Cast the input to text, drop records without a group, apply filters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_compare.models import MISSING_TEXT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itk_compare.models import AttributeFilter, CompareConfig


def check_columns(frame: pl.DataFrame, config: CompareConfig) -> None:
    """Raise ``KeyError`` naming any configured column the frame lacks."""
    missing = [c for c in config.required_columns() if c not in frame.columns]
    if missing:
        msg = f"Columns not in the data: {missing}"
        raise KeyError(msg)


def prepare_frame(frame: pl.DataFrame, groups: Sequence[str]) -> pl.DataFrame:
    """Every column as text with ``""`` read as missing; rows lacking a group value dropped.

    Casting once here means every later stage compares the text the user
    uploaded, which is what the legacy melt-to-string did implicitly.
    """
    text = frame.with_columns([
        pl.when(pl.col(c).cast(pl.Utf8) == "")
        .then(None)
        .otherwise(pl.col(c).cast(pl.Utf8))
        .alias(c)
        for c in frame.columns
    ])
    return text.drop_nulls(subset=list(groups))


def apply_filters(
    frame: pl.DataFrame, filters: Sequence[AttributeFilter]
) -> pl.DataFrame:
    """Keep records matching every filter (text equality)."""
    for item in filters:
        frame = frame.filter(pl.col(item.attribute) == item.value)
    return frame


def filter_options(frame: pl.DataFrame) -> list[str]:
    """Every ``column:value`` pair in the data, sorted, skipping missing values."""
    options: list[str] = []
    for column in sorted(frame.columns):
        values = (
            frame.select(pl.col(column).cast(pl.Utf8).unique())
            .to_series()
            .drop_nulls()
            .sort()
        )
        options.extend(f"{column}:{v}" for v in values if v not in MISSING_TEXT)
    return options
