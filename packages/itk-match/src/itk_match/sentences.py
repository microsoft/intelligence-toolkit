# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Render unified records as text for embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from itk_match.models import NAME
from itk_match.unify import attribute_columns

if TYPE_CHECKING:
    import polars as pl


def to_sentence(name: str, attributes: dict[str, str | None]) -> str:
    parts = [f"NAME: {name.upper()}"]
    parts.extend(
        f"{label.upper()}: {value.upper()}"
        for label, value in attributes.items()
        if value is not None and value != ""
    )
    return "; ".join(parts)


def to_sentences(records: pl.DataFrame) -> list[str]:
    """One sentence per row, in row order, from the name plus non-null attributes."""
    labels = attribute_columns(records)
    return [
        to_sentence(row[NAME], {label: row[label] for label in labels})
        for row in records.iter_rows(named=True)
    ]
