# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed selections of attribute values and the data schema they draw from."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from itk_anonymize.models import AttributeValue
from itk_anonymize.prepare import EMPTY

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def selection_from_dicts(items: Iterable[Mapping[str, str]]) -> list[AttributeValue]:
    """Build a selection from the ``{"attribute": ..., "value": ...}`` dicts the app uses."""
    return [
        AttributeValue(attribute=item["attribute"], value=item["value"])
        for item in items
    ]


def selection_keys(
    selection: Sequence[AttributeValue], value_separator: str = ":"
) -> list[str]:
    return [item.key(value_separator) for item in selection]


def describe_selection(
    selection: Sequence[AttributeValue], multiline: bool = True
) -> str:
    """Human-readable summary, as the legacy ``print_selections``.

    Multiline: one ``- attribute = v1 | v2`` line per attribute. Single line:
    ``attribute:v1|v2`` terms joined by commas. Values are sorted within an
    attribute; attributes keep first-seen order.
    """
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for item in selection:
        grouped[item.attribute].append(item.value)
    for values in grouped.values():
        values.sort()
    if multiline:
        return "".join(
            f"- {attribute} = " + " | ".join(values) + "\n"
            for attribute, values in grouped.items()
        )
    return ", ".join(
        f"{attribute}:" + "|".join(values) for attribute, values in grouped.items()
    )


def data_schema(frame: pl.DataFrame | None) -> dict[str, list[str]]:
    """Sorted distinct non-empty text values per column."""
    schema: dict[str, list[str]] = {}
    if frame is None:
        return schema
    for name in frame.columns:
        values = frame.get_column(name).cast(pl.Utf8).fill_null(EMPTY).unique()
        schema[name] = sorted(v for v in values.to_list() if v != EMPTY)
    return schema
