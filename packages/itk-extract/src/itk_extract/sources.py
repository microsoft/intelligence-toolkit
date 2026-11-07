# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build ``Source`` lists from plain texts or table rows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from itk_extract.models import Source

if TYPE_CHECKING:
    import polars as pl


def from_texts(texts: list[str], prefix: str = "text") -> list[Source]:
    """One source per text, with ids ``prefix-0``, ``prefix-1``, ..."""
    return [Source(id=f"{prefix}-{i}", text=text) for i, text in enumerate(texts)]


def from_frame(
    frame: pl.DataFrame,
    columns: list[str] | None = None,
    id_column: str | None = None,
    prefix: str = "row",
) -> list[Source]:
    """One source per row, with the row rendered as a JSON object.

    ``columns`` restricts which columns are rendered (all by default). When
    ``id_column`` is given its values become the source ids; otherwise ids are
    ``prefix-<row index>``. The full row is kept in ``Source.meta``.
    """
    wanted = columns or frame.columns
    missing = sorted(set(wanted) - set(frame.columns))
    if missing:
        msg = f"Frame has no columns {missing}"
        raise ValueError(msg)
    if id_column is not None and id_column not in frame.columns:
        msg = f"Frame has no id column {id_column!r}"
        raise ValueError(msg)
    sources: list[Source] = []
    for i, row in enumerate(frame.iter_rows(named=True)):
        source_id = str(row[id_column]) if id_column is not None else f"{prefix}-{i}"
        text = json.dumps({c: row[c] for c in wanted}, ensure_ascii=False, default=str)
        sources.append(Source(id=source_id, text=text, meta=dict(row)))
    check_unique_ids(sources)
    return sources


def check_unique_ids(sources: list[Source]) -> None:
    seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            msg = f"Duplicate source id {source.id!r}"
            raise ValueError(msg)
        seen.add(source.id)
