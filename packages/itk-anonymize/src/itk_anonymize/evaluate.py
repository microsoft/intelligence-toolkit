# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Accuracy of one aggregate set against another, per combination length."""

from __future__ import annotations

from collections import defaultdict

from itk_anonymize.models import Aggregates, ErrorReport, ErrorRow


def error_report(source: Aggregates, target: Aggregates) -> ErrorReport:
    """Mean count, mean absolute error, suppressed and fabricated percentages.

    One row per combination length that both sets contain, then an overall
    row. Means over nothing are ``None`` rather than NaN; percentages over a
    zero total are ``None`` rather than an exception.
    """
    source_total, source_total_by_len = _totals(source)
    target_total, target_total_by_len = _totals(target)

    counts: list[int] = []
    counts_by_len: defaultdict[int, list[int]] = defaultdict(list)
    errors: list[int] = []
    errors_by_len: defaultdict[int, list[int]] = defaultdict(list)
    suppressed = 0
    suppressed_by_len: defaultdict[int, int] = defaultdict(int)
    for key, count in source.items():
        counts.append(count)
        counts_by_len[len(key)].append(count)
        if key in target:
            error = abs(target[key] - count)
            errors.append(error)
            errors_by_len[len(key)].append(error)
        else:
            suppressed += count
            suppressed_by_len[len(key)] += count

    fabricated = 0
    fabricated_by_len: defaultdict[int, int] = defaultdict(int)
    for key, count in target.items():
        if key not in source:
            fabricated += count
            fabricated_by_len[len(key)] += count

    rows = [
        ErrorRow(
            length=length,
            mean_count=_mean(counts_by_len[length]),
            mean_error=_mean(errors_by_len[length]),
            suppressed_pct=_pct(suppressed_by_len[length], source_total_by_len[length]),
            fabricated_pct=_pct(fabricated_by_len[length], target_total_by_len[length]),
        )
        for length in sorted(errors_by_len)
    ]
    rows.append(
        ErrorRow(
            length=None,
            mean_count=_mean(counts),
            mean_error=_mean(errors),
            suppressed_pct=_pct(suppressed, source_total),
            fabricated_pct=_pct(fabricated, target_total),
        )
    )
    return ErrorReport(tuple(rows))


def _totals(aggregates: Aggregates) -> tuple[int, defaultdict[int, int]]:
    total = 0
    by_len: defaultdict[int, int] = defaultdict(int)
    for key, count in aggregates.items():
        total += count
        by_len[len(key)] += count
    return total, by_len


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _pct(part: int, total: int) -> float | None:
    return part * 100.0 / total if total else None
