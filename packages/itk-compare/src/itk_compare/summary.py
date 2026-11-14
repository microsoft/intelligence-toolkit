# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Join the stage frames into the summary, describe it, and pick report rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_compare.models import (
    ATTRIBUTE_VALUE,
    GROUP_RANK,
    CompareConfig,
    GroupKey,
    ReportSelection,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_summary(
    config: CompareConfig,
    groups_frame: pl.DataFrame,
    attributes_frame: pl.DataFrame,
    temporal_frame: pl.DataFrame | None,
) -> pl.DataFrame:
    """One row per group and attribute value (and window, when temporal).

    Sorted by group columns, attribute value and window, so equal inputs
    give identical frames.
    """
    groups = list(config.groups)
    keys = [*groups, ATTRIBUTE_VALUE]
    windows = config.window_columns
    if windows is not None:
        if temporal_frame is None:
            msg = "config has a temporal column but no temporal frame was given"
            raise ValueError(msg)
        base = temporal_frame.join(groups_frame, on=groups, how="left").join(
            attributes_frame, on=keys, how="left"
        )
        order = [*keys, windows.window]
    else:
        base = attributes_frame.join(groups_frame, on=groups, how="left")
        order = keys
    return base.select(config.summary_columns).sort(order)


def _format_list(items: Sequence[str], escape_colon: bool = False) -> str:
    formatted = []
    for item in items:
        text = item.replace(":", "\\:") if escape_colon else item
        formatted.append(f"**{text}**")
    return "[" + ", ".join(formatted) + "]"


def describe_summary(
    config: CompareConfig, filtered_count: int, dataset_proportion: int
) -> str:
    """The markdown paragraph the legacy workflow showed under the summary."""
    groups_text = _format_list(config.groups)
    lines = ["This table shows:"]
    if config.filters:
        filters_text = _format_list([f.key for f in config.filters], escape_colon=True)
        lines.append(
            f"- A summary of **{filtered_count}** data records matching {filters_text}, "
            f"representing **{dataset_proportion}%** of the overall dataset with values "
            "for all grouping attributes"
        )
    else:
        lines.append(
            f"- A summary of all **{filtered_count}** data records with values for all "
            "grouping attributes"
        )
    lines.extend([
        f"- The **group_count** of records for all {groups_text} groups, and "
        "corresponding **group_rank**",
        f"- The **group_attribute_count** of each **attribute_value** for all "
        f"{groups_text} groups, and corresponding **group_attribute_rank**",
    ])
    temporal = config.temporal
    if temporal is not None:
        lines.extend([
            f"- The **{temporal}_window_count** of each **attribute_value** for each "
            f"**{temporal}_window** for all {groups_text} groups, and corresponding "
            f"**{temporal}_window_rank**",
            f"- The **{temporal}_window_delta**, or change in the "
            f"**attribute_value_count** for successive **{temporal}_window** values, "
            f"within each {groups_text} group",
        ])
    return "\n".join(lines)


def group_options(summary: pl.DataFrame, groups: Sequence[str]) -> list[GroupKey]:
    """Every group in the summary as a column -> value dict, sorted."""
    keys = summary.select(list(groups)).unique().to_dicts()
    return sorted(keys, key=lambda k: tuple(k[g] for g in groups))


def select_report_rows(
    summary: pl.DataFrame, selection: ReportSelection
) -> tuple[pl.DataFrame, str]:
    """The summary rows a report covers, and the sentence describing that choice."""
    if selection.groups:
        expr = pl.lit(False)
        for key in selection.groups:
            group_expr = pl.lit(True)
            for column, value in key.items():
                group_expr = group_expr & (pl.col(column) == value)
            expr = expr | group_expr
        described = ", ".join(str(k) for k in selection.groups)
        return summary.filter(expr), (
            f"Filtered to the following groups only: {described}"
        )
    if selection.top_group_ranks is not None:
        top = selection.top_group_ranks
        return summary.filter(pl.col(GROUP_RANK) <= top), (
            f"Filtered to the top {top} groups by record count"
        )
    return summary, ""
