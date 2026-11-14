# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Align several datasets onto one long frame with canonical columns."""

from __future__ import annotations

import polars as pl

from itk_match.models import (
    DATASET,
    ENTITY_ID,
    NAME,
    RECORD_ID,
    RESERVED_COLUMNS,
    Dataset,
)


def _unify_one(dataset: Dataset, max_rows: int | None) -> pl.DataFrame:
    frame = dataset.frame
    if dataset.id_column is None:
        frame = frame.with_row_index(ENTITY_ID)
        id_expr = pl.col(ENTITY_ID)
    else:
        id_expr = pl.col(dataset.id_column)

    selected = frame.select(
        pl.lit(dataset.name).alias(DATASET),
        id_expr.cast(pl.Utf8).alias(ENTITY_ID),
        pl.col(dataset.name_column).cast(pl.Utf8).str.strip_chars().alias(NAME),
        *[
            pl.col(source).cast(pl.Utf8).str.strip_chars().alias(label)
            for source, label in dataset.attributes.items()
        ],
    )
    selected = selected.filter(pl.col(NAME).is_not_null() & (pl.col(NAME) != ""))
    if max_rows is not None:
        selected = selected.head(max_rows)
    return selected


def unify(
    datasets: list[Dataset], max_rows_per_dataset: int | None = None
) -> pl.DataFrame:
    """Return one frame with columns record_id, dataset, entity_id, name, labels.

    Every value is a string. Labels missing from a dataset are null for its
    rows. Rows with an empty name are dropped because the name drives both the
    sentence and the lexical score.
    """
    if not datasets:
        return pl.DataFrame(schema=dict.fromkeys(RESERVED_COLUMNS, pl.Utf8))
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        msg = f"Dataset names must be unique, got {names}"
        raise ValueError(msg)

    parts = [_unify_one(d, max_rows_per_dataset) for d in datasets]
    frame = pl.concat(parts, how="diagonal")
    labels = sorted({label for d in datasets for label in d.labels})
    return frame.with_columns(
        (pl.col(DATASET) + ":" + pl.col(ENTITY_ID)).alias(RECORD_ID)
    ).select([RECORD_ID, DATASET, ENTITY_ID, NAME, *labels])


def attribute_columns(records: pl.DataFrame) -> list[str]:
    """The canonical attribute labels present in a unified frame."""
    return [c for c in records.columns if c not in RESERVED_COLUMNS]
