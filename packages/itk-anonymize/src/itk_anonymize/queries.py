# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Counts over synthetic rows and DP aggregates for charts and summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_anonymize.aggregate import normalize_key
from itk_anonymize.models import (
    AGGREGATE,
    ATTRIBUTE,
    ATTRIBUTE_VALUE,
    COUNT,
    DATASET,
    HIGHLIGHT,
    PROPORTION,
    SOURCE,
    SYNTHETIC,
    TARGET,
    Aggregates,
    AttributeValue,
)
from itk_anonymize.prepare import EMPTY
from itk_anonymize.selection import data_schema

if TYPE_CHECKING:
    from collections.abc import Sequence

COUNTS_SCHEMA = {
    ATTRIBUTE: pl.Utf8,
    ATTRIBUTE_VALUE: pl.Utf8,
    COUNT: pl.Int64,
    DATASET: pl.Utf8,
}
LINKS_SCHEMA = {
    SOURCE: pl.Utf8,
    TARGET: pl.Utf8,
    COUNT: pl.Int64,
    HIGHLIGHT: pl.Int64,
    PROPORTION: pl.Float64,
    DATASET: pl.Utf8,
}


def top_attributes(
    synthetic: pl.DataFrame,
    aggregates: Aggregates,
    selection: Sequence[AttributeValue] = (),
    show_attributes: Sequence[str] = (),
    num_values: int = 0,
    value_separator: str = ":",
) -> pl.DataFrame:
    """Attribute value counts among records matching ``selection``.

    Counts come from the synthetic rows, replaced by the exact DP aggregate
    count wherever the selection plus that value is a counted combination.
    Several values of one attribute in the selection act as a union, in which
    case only synthetic counts are reported (as in the legacy workflow).
    ``num_values`` of 0 keeps every row. Sorted by count descending, then
    attribute value ascending.
    """
    schema = data_schema(synthetic)
    wanted = {(item.attribute, item.value) for item in selection}
    filtered = synthetic
    single_terms: list[str] = []
    has_unions = False
    for attribute, values in schema.items():
        chosen = [v for v in values if (attribute, v) in wanted]
        if not chosen:
            continue
        filtered = filtered.filter(pl.col(attribute).is_in(chosen))
        if len(chosen) == 1:
            single_terms.append(f"{attribute}{value_separator}{chosen[0]}")
        else:
            has_unions = True

    result = _synthetic_counts(filtered, value_separator)
    if not has_unions:
        rows = []
        selected = set(single_terms)
        for attribute, values in schema.items():
            for value in values:
                term = f"{attribute}{value_separator}{value}"
                key = normalize_key(
                    single_terms if term in selected else [*single_terms, term]
                )
                count = aggregates.get(key, 0)
                if count > 0:
                    rows.append((attribute, term, count, AGGREGATE))
        exact = _frame(rows, COUNTS_SCHEMA)
        result = pl.concat([
            result.filter(
                ~pl.col(ATTRIBUTE_VALUE).is_in(exact.get_column(ATTRIBUTE_VALUE))
            ),
            exact,
        ])
    if show_attributes:
        result = result.filter(pl.col(ATTRIBUTE).is_in(list(show_attributes)))
    result = result.sort([COUNT, ATTRIBUTE_VALUE], descending=[True, False])
    if num_values > 0:
        result = result.head(num_values)
    return result


def _synthetic_counts(frame: pl.DataFrame, value_separator: str) -> pl.DataFrame:
    if frame.width == 0 or frame.height == 0:
        return _frame([], COUNTS_SCHEMA)
    long = (
        frame.cast(pl.Utf8)
        .fill_null(EMPTY)
        .melt(variable_name=ATTRIBUTE, value_name="value")
        .filter(pl.col("value") != EMPTY)
    )
    return (
        long.group_by(ATTRIBUTE, "value")
        .agg(pl.len().cast(pl.Int64).alias(COUNT))
        .select(
            pl.col(ATTRIBUTE),
            (pl.col(ATTRIBUTE) + pl.lit(value_separator) + pl.col("value")).alias(
                ATTRIBUTE_VALUE
            ),
            pl.col(COUNT),
            pl.lit(SYNTHETIC).alias(DATASET),
        )
    )


def time_series(
    synthetic: pl.DataFrame,
    aggregates: Aggregates,
    selection: Sequence[AttributeValue],
    time_attribute: str,
    series_attributes: Sequence[str],
    value_separator: str = ":",
) -> pl.DataFrame:
    """``top_attributes`` for each distinct time value, as one long frame.

    Attribute values missing at some time are filled with a zero count tagged
    ``Aggregate`` (legacy behaviour). Sorted by time then attribute value.
    """
    _require_columns(synthetic, [time_attribute])
    times = sorted(
        t
        for t in synthetic.get_column(time_attribute).cast(pl.Utf8).unique().to_list()
        if t is not None and t != EMPTY
    )
    schema = {time_attribute: pl.Utf8, **COUNTS_SCHEMA}
    frames = []
    for time in times:
        counts = top_attributes(
            synthetic,
            aggregates,
            [*selection, AttributeValue(attribute=time_attribute, value=time)],
            series_attributes,
            0,
            value_separator,
        )
        frames.append(
            counts.with_columns(pl.lit(time).alias(time_attribute)).select(schema)
        )
    if not frames:
        return _frame([], schema)
    present = pl.concat(frames)
    pairs = present.select(ATTRIBUTE, ATTRIBUTE_VALUE).unique()
    grid = pairs.join(pl.DataFrame({time_attribute: times}), how="cross")
    missing = grid.join(
        present.select(time_attribute, ATTRIBUTE, ATTRIBUTE_VALUE),
        on=[time_attribute, ATTRIBUTE, ATTRIBUTE_VALUE],
        how="anti",
    ).with_columns(
        pl.lit(0, dtype=pl.Int64).alias(COUNT), pl.lit(AGGREGATE).alias(DATASET)
    )
    return (
        pl.concat([present, missing.select(schema)])
        .unique(maintain_order=True)
        .sort([time_attribute, ATTRIBUTE_VALUE, DATASET, COUNT])
    )


def aggregate_links(
    aggregates: Aggregates,
    filters: Sequence[AttributeValue],
    source_attribute: str,
    target_attribute: str,
    highlight: AttributeValue | None = None,
    value_separator: str = ":",
) -> pl.DataFrame:
    """Source/target value pairs and their exact DP counts under ``filters``.

    A combination contributes when it holds both edge attributes, every filter
    term, and nothing else; one extra term equal to ``highlight`` contributes
    to that edge's highlight count instead.
    """
    edge_attributes = {source_attribute, target_attribute}
    filter_terms = {f.key(value_separator) for f in filters}
    highlight_term = highlight.key(value_separator) if highlight else None
    counts: dict[tuple[str, str], int] = {}
    highlights: dict[tuple[str, str], int] = {}
    for key, count in aggregates.items():
        terms = [t.partition(value_separator) for t in key]
        attributes = {attribute for attribute, _, _ in terms}
        if not edge_attributes <= attributes or not filter_terms <= set(key):
            continue
        source_value = target_value = None
        remaining: list[str] = []
        for attribute, _, value in terms:
            term = f"{attribute}{value_separator}{value}"
            if attribute == source_attribute:
                source_value = value
            if attribute == target_attribute:
                target_value = value
            if (
                attribute not in edge_attributes
                and term not in filter_terms
                and term not in remaining
            ):
                remaining.append(term)
        edge = (str(source_value), str(target_value))
        if not remaining:
            counts[edge] = count
        elif len(remaining) == 1 and remaining[0] == highlight_term:
            highlights[edge] = count
    rows = [
        (
            s,
            t,
            count,
            highlights.get((s, t), 0),
            highlights.get((s, t), 0) / count,
            AGGREGATE,
        )
        for (s, t), count in counts.items()
        if count > 0
    ]
    return _links_frame(rows)


def synthetic_links(
    synthetic: pl.DataFrame,
    filters: Sequence[AttributeValue],
    source_attribute: str,
    target_attribute: str,
    highlight: AttributeValue | None = None,
) -> pl.DataFrame:
    """Source/target value pairs counted over synthetic rows under ``filters``.

    Filter terms on the same attribute are a union; different attributes are
    an intersection. Pairs with an empty source or target are skipped.
    """
    needed = [source_attribute, target_attribute, *(f.attribute for f in filters)]
    if highlight is not None:
        needed.append(highlight.attribute)
    _require_columns(synthetic, needed)
    frame = synthetic.cast(pl.Utf8).fill_null(EMPTY)
    frame = frame.filter(
        (pl.col(source_attribute) != EMPTY) & (pl.col(target_attribute) != EMPTY)
    )
    groups: dict[str, list[str]] = {}
    for item in filters:
        groups.setdefault(item.attribute, []).append(item.value)
    for attribute, values in groups.items():
        frame = frame.filter(pl.col(attribute).is_in(values))
    highlight_expr = (
        (pl.col(highlight.attribute) == highlight.value).cast(pl.Int64).sum()
        if highlight is not None
        else pl.lit(0, dtype=pl.Int64)
    )
    counted = (
        frame.group_by(source_attribute, target_attribute)
        .agg(pl.len().cast(pl.Int64).alias(COUNT), highlight_expr.alias(HIGHLIGHT))
        .sort([source_attribute, target_attribute])
    )
    rows = [(s, t, c, h, h / c, SYNTHETIC) for s, t, c, h in counted.rows()]
    return _links_frame(rows)


def _links_frame(rows: list[tuple]) -> pl.DataFrame:
    return _frame(rows, LINKS_SCHEMA)


def _frame(rows: list[tuple], schema: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    """Frame from row tuples; an empty frame still carries the schema."""
    if not rows:
        return pl.DataFrame(schema=schema)
    columns = zip(*rows, strict=True)
    return pl.DataFrame(
        {name: list(values) for name, values in zip(schema, columns, strict=True)},
        schema=schema,
    )


def _require_columns(frame: pl.DataFrame, names: Sequence[str]) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        msg = f"Frame has no columns {missing}"
        raise ValueError(msg)
