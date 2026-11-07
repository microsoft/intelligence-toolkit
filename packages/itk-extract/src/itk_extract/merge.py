# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Combine per-chunk objects into one, keeping track of what disagreed."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from itk_extract.models import CHUNK_ID, SOURCE_ID, Conflict

if TYPE_CHECKING:
    from itk_extract.models import Chunk, Extraction, MergePolicy

_Item = tuple["Chunk", Any]


def merge(
    extractions: list[Extraction],
    policy: MergePolicy = "last",
    annotate_provenance: bool = True,
) -> tuple[dict[str, Any], list[Conflict]]:
    """Merge extracted objects field by field.

    - Objects merge recursively.
    - Arrays concatenate in extraction order. With ``annotate_provenance``
      each object element gains ``source_id`` and ``chunk_id`` fields.
    - Scalars that agree (ignoring ``None`` and empty strings) are kept.
      Scalars that disagree are resolved by ``policy`` and reported as a
      ``Conflict``; ``keep_all`` keeps every distinct value as a list.
    """
    conflicts: list[Conflict] = []
    items: list[_Item] = [(e.chunk, e.data) for e in extractions]
    merged = _merge_dicts(items, (), policy, annotate_provenance, conflicts)
    return merged, conflicts


def _merge_dicts(
    items: list[_Item],
    path: tuple[str, ...],
    policy: MergePolicy,
    annotate: bool,
    conflicts: list[Conflict],
) -> dict[str, Any]:
    keys: dict[str, None] = {}
    for _, data in items:
        keys.update(dict.fromkeys(data))
    out: dict[str, Any] = {}
    for key in keys:
        present = [(chunk, data[key]) for chunk, data in items if key in data]
        out[key] = _merge_values(present, (*path, key), policy, annotate, conflicts)
    return out


def _merge_values(
    present: list[_Item],
    path: tuple[str, ...],
    policy: MergePolicy,
    annotate: bool,
    conflicts: list[Conflict],
) -> Any:
    values = [value for _, value in present]
    if values and all(isinstance(v, dict) for v in values):
        return _merge_dicts(present, path, policy, annotate, conflicts)
    if values and all(isinstance(v, list) for v in values):
        return _concat_lists(present, annotate)

    non_empty = [(c, v) for c, v in present if not _is_empty(v)]
    if not non_empty:
        return values[-1] if values else None
    distinct = _distinct(v for _, v in non_empty)
    if len(distinct) == 1:
        return distinct[0]
    chosen = _resolve(policy, non_empty, distinct)
    conflicts.append(
        Conflict(".".join(path), chosen, tuple((c.id, v) for c, v in non_empty))
    )
    return chosen


def _concat_lists(present: list[_Item], annotate: bool) -> list[Any]:
    out: list[Any] = []
    for chunk, elements in present:
        for element in elements:
            if annotate and isinstance(element, dict):
                out.append({**element, SOURCE_ID: chunk.source_id, CHUNK_ID: chunk.id})
            else:
                out.append(element)
    return out


def _resolve(policy: MergePolicy, non_empty: list[_Item], distinct: list[Any]) -> Any:
    if policy == "first":
        return non_empty[0][1]
    if policy == "keep_all":
        return distinct
    if policy == "majority":
        keys = [_key(v) for _, v in non_empty]
        counts = Counter(keys)
        best = max(counts.values())
        winner = next(k for k in reversed(keys) if counts[k] == best)
        return next(v for _, v in non_empty if _key(v) == winner)
    return non_empty[-1][1]


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _distinct(values: Any) -> list[Any]:
    seen: dict[str, Any] = {}
    for value in values:
        seen.setdefault(_key(value), value)
    return list(seen.values())
