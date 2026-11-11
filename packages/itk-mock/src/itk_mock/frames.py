# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Turn the record arrays of a generated object into flat polars frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

VALUE = "value"


def records_at(data: Any, record_path: Sequence[str]) -> list[Any]:
    """Elements of the array found by following ``record_path``.

    Arrays met along the way are fanned out, so ``["people", "events"]``
    yields every event of every person.
    """
    if not record_path:
        return list(data) if isinstance(data, list) else []
    if isinstance(data, dict):
        return records_at(data.get(record_path[0]), record_path[1:])
    if isinstance(data, list):
        return [r for element in data for r in records_at(element, record_path)]
    return []


def flatten(element: Any) -> dict[str, Any]:
    """One flat row: nested objects become ``outer.inner`` keys, lists stay."""
    if not isinstance(element, dict):
        return {VALUE: element}
    row: dict[str, Any] = {}

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, name)
            else:
                row[name] = value

    walk(element, "")
    return row


def extract_frame(data: Any, record_path: Sequence[str]) -> pl.DataFrame:
    """Flat frame of the array at ``record_path``; no records give no columns."""
    rows = [flatten(element) for element in records_at(data, record_path)]
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows, infer_schema_length=None)


def to_frames(
    data: Any, record_arrays: Sequence[Sequence[str]]
) -> dict[str, pl.DataFrame]:
    """One frame per record array, keyed by its dotted label path."""
    return {".".join(path): extract_frame(data, path) for path in record_arrays}
