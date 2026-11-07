# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Turn the record arrays of a merged object into flat polars frames."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl

from itk_extract.models import PROVENANCE_COLUMNS

if TYPE_CHECKING:
    from itk_extract.schema import RecordSchema

VALUE = "value"


def records_at(data: Any, path: tuple[str, ...]) -> list[Any]:
    """Elements of the array found by following ``path``.

    Arrays met along the way are fanned out, so ``("people", "events")``
    yields every event of every person.
    """
    if not path:
        return list(data) if isinstance(data, list) else []
    head, *rest = path
    if isinstance(data, dict):
        return records_at(data.get(head), (*rest,))
    if isinstance(data, list):
        return [r for element in data for r in records_at(element, path)]
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


def records_frame(
    data: Any, path: tuple[str, ...], dedupe: bool = False
) -> pl.DataFrame:
    """Flat frame of the array at ``path``; provenance columns last."""
    rows = [flatten(e) for e in records_at(data, path)]
    if dedupe:
        rows = _dedupe(rows)
    if not rows:
        return pl.DataFrame()
    frame = pl.from_dicts(rows, infer_schema_length=None)
    ordered = [c for c in frame.columns if c not in PROVENANCE_COLUMNS] + [
        c for c in PROVENANCE_COLUMNS if c in frame.columns
    ]
    return frame.select(ordered)


def to_frames(
    data: Any, schema: RecordSchema, dedupe: bool = False
) -> dict[str, pl.DataFrame]:
    """One frame per record array in the schema, keyed by its dotted label."""
    return {
        ".".join(path): records_frame(data, path, dedupe) for path in schema.array_paths
    }


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        content = {k: v for k, v in row.items() if k not in PROVENANCE_COLUMNS}
        key = json.dumps(content, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept
