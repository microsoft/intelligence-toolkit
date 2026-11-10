# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs and outputs of the case group comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import polars as pl

# Summary frame columns, identical to the legacy workflow so downstream code
# built against the summary CSV keeps working.
GROUP_COUNT = "group_count"
GROUP_RANK = "group_rank"
ATTRIBUTE_VALUE = "attribute_value"
GROUP_ATTRIBUTE_COUNT = "group_attribute_count"
GROUP_ATTRIBUTE_RANK = "group_attribute_rank"

# Text values that count as missing when listing filter options.
MISSING_TEXT = frozenset({"", "<NA>", "nan", "NaN", "None", "none", "NULL", "null"})

# A group key: group column name -> value, as the summary frame holds it.
GroupKey = dict[str, str]


class WindowColumns(NamedTuple):
    """The four temporal columns, named after the temporal attribute."""

    window: str
    count: str
    rank: str
    delta: str


def window_columns(temporal: str) -> WindowColumns:
    return WindowColumns(
        f"{temporal}_window",
        f"{temporal}_window_count",
        f"{temporal}_window_rank",
        f"{temporal}_window_delta",
    )


class AttributeFilter(BaseModel):
    """One ``attribute = value`` condition records must satisfy."""

    model_config = ConfigDict(frozen=True)

    attribute: str = Field(min_length=1)
    value: str

    @property
    def key(self) -> str:
        return f"{self.attribute}:{self.value}"

    @classmethod
    def parse(cls, text: str) -> AttributeFilter:
        """Parse ``attribute:value``; the value may itself contain colons."""
        attribute, separator, value = text.partition(":")
        if not separator:
            msg = f"{text!r} has no ':' between attribute and value"
            raise ValueError(msg)
        return cls(attribute=attribute, value=value)


class CompareConfig(BaseModel):
    """Which columns define groups, which are counted, and how records are filtered."""

    model_config = ConfigDict(frozen=True)

    groups: tuple[str, ...] = Field(
        min_length=1, description="Columns whose value combinations define groups"
    )
    aggregates: tuple[str, ...] = Field(
        min_length=1, description="Columns whose values are counted per group"
    )
    temporal: str | None = Field(
        None, description="Optional temporal/ordinal column defining windows"
    )
    filters: tuple[AttributeFilter, ...] = Field(
        (), description="Records must match every filter before summarising"
    )

    @model_validator(mode="after")
    def _check_columns(self) -> CompareConfig:
        for name, columns in (("groups", self.groups), ("aggregates", self.aggregates)):
            if len(set(columns)) != len(columns):
                msg = f"{name} must not repeat a column"
                raise ValueError(msg)
        if self.temporal is not None:
            if self.temporal in self.groups:
                msg = f"temporal column {self.temporal!r} cannot also be a group"
                raise ValueError(msg)
            if self.temporal in self.aggregates:
                msg = f"temporal column {self.temporal!r} cannot also be an aggregate"
                raise ValueError(msg)
        return self

    @property
    def window_columns(self) -> WindowColumns | None:
        return window_columns(self.temporal) if self.temporal is not None else None

    @property
    def summary_columns(self) -> list[str]:
        """Column order of the summary frame."""
        columns = [
            *self.groups,
            GROUP_COUNT,
            GROUP_RANK,
            ATTRIBUTE_VALUE,
            GROUP_ATTRIBUTE_COUNT,
            GROUP_ATTRIBUTE_RANK,
        ]
        windows = self.window_columns
        if windows is not None:
            columns.extend(windows)
        return columns

    def required_columns(self) -> list[str]:
        """Every input column the config refers to, in a stable order."""
        columns = [*self.groups, *self.aggregates]
        if self.temporal is not None:
            columns.append(self.temporal)
        for item in self.filters:
            if item.attribute not in columns:
                columns.append(item.attribute)
        return columns


class ReportSelection(BaseModel):
    """Which summary rows go into a report: named groups, or the top ranks.

    With neither set, the whole summary is reported. Named groups win over
    ``top_group_ranks`` when both are given, as in the legacy workflow.
    """

    model_config = ConfigDict(frozen=True)

    groups: tuple[GroupKey, ...] = Field(
        (), description="Group keys (group column -> value) to report on"
    )
    top_group_ranks: int | None = Field(
        None, ge=1, description="Report on groups ranked at or above this"
    )

    @property
    def is_empty(self) -> bool:
        return not self.groups and self.top_group_ranks is None


@dataclass(slots=True)
class CompareResult:
    """Everything produced by one comparison run.

    Intermediate frames are kept so report selections and descriptions are
    answered from them without re-running the pipeline.
    """

    config: CompareConfig
    prepared: pl.DataFrame
    filtered: pl.DataFrame
    groups_frame: pl.DataFrame
    attributes_frame: pl.DataFrame
    temporal_frame: pl.DataFrame | None
    summary: pl.DataFrame

    @property
    def dataset_proportion(self) -> int:
        """Whole percentage of prepared records that passed the filters."""
        if self.prepared.height == 0:
            return 0
        return round(100 * self.filtered.height / self.prepared.height)

    @property
    def description(self) -> str:
        from itk_compare.summary import describe_summary

        return describe_summary(
            self.config, self.filtered.height, self.dataset_proportion
        )
