# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Attribute combination counts, in pacsynth's key format but order-insensitive."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

import polars as pl

from itk_anonymize.models import (
    PROTECTED_COUNT,
    RECORD_COUNT,
    SELECTIONS,
    Aggregates,
    CombinationKey,
    RawTable,
)
from itk_anonymize.prepare import EMPTY

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ZERO = "0"


def normalize_key(parts: Iterable[str]) -> CombinationKey:
    """Canonical form of a combination: its parts sorted lexicographically."""
    return tuple(sorted(parts))


def render_key(key: CombinationKey, attribute_separator: str = ";") -> str:
    return attribute_separator.join(key)


def record_attributes(
    header: Sequence[str],
    row: Sequence[str],
    value_separator: str = ":",
    sensitive_zeros: Sequence[str] = (),
) -> list[str]:
    """The ``attribute:value`` parts of one record that pacsynth would count.

    Empty values are never attributes; ``"0"`` only is for ``sensitive_zeros``.
    """
    zeros = set(sensitive_zeros)
    return [
        f"{name}{value_separator}{value}"
        for name, value in zip(header, row, strict=True)
        if value != EMPTY and (value != ZERO or name in zeros)
    ]


def count_combinations(
    table: RawTable,
    reporting_length: int,
    value_separator: str = ":",
    sensitive_zeros: Sequence[str] = (),
) -> Aggregates:
    """Plain (non-private) counts of every combination up to ``reporting_length``.

    Mirrors ``pacsynth.Dataset.get_aggregates``; the step-3 comparison checks
    the two agree.
    """
    if reporting_length < 1:
        msg = "reporting_length must be at least 1"
        raise ValueError(msg)
    counts: Counter[CombinationKey] = Counter()
    for row in table.rows:
        parts = sorted(
            record_attributes(table.header, row, value_separator, sensitive_zeros)
        )
        for length in range(1, min(reporting_length, len(parts)) + 1):
            counts.update(combinations(parts, length))
    return dict(counts)


def parse_aggregates(raw: dict[str, int], attribute_separator: str = ";") -> Aggregates:
    """Engine output (``"A:a;B:b" -> n``) as normalized keys.

    Keys that coincide after normalization have their counts summed.
    """
    parsed: Counter[CombinationKey] = Counter()
    for text, count in raw.items():
        parsed[normalize_key(text.split(attribute_separator))] += count
    return dict(parsed)


def aggregates_frame(
    aggregates: Aggregates,
    protected_record_count: int,
    attribute_separator: str = ";",
) -> pl.DataFrame:
    """``selections`` / ``protected_count`` frame with a ``record_count`` row.

    Sorted by count descending then key ascending, so output is deterministic.
    """
    frame = pl.DataFrame(
        {
            SELECTIONS: [render_key(k, attribute_separator) for k in aggregates],
            PROTECTED_COUNT: list(aggregates.values()),
        },
        schema={SELECTIONS: pl.Utf8, PROTECTED_COUNT: pl.Int64},
    )
    record_row = pl.DataFrame(
        {SELECTIONS: [RECORD_COUNT], PROTECTED_COUNT: [protected_record_count]},
        schema={SELECTIONS: pl.Utf8, PROTECTED_COUNT: pl.Int64},
    )
    return pl.concat([frame, record_row]).sort(
        [PROTECTED_COUNT, SELECTIONS], descending=[True, False]
    )
