# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs and outputs of the anonymization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Aggregate frame columns (Synthetic Data Showcase file format).
SELECTIONS = "selections"
PROTECTED_COUNT = "protected_count"
RECORD_COUNT = "record_count"

# Query frame columns, kept identical to the legacy workflow so chart code
# built against it keeps working.
ATTRIBUTE = "Attribute"
ATTRIBUTE_VALUE = "Attribute Value"
COUNT = "Count"
DATASET = "Dataset"
SOURCE = "Source"
TARGET = "Target"
HIGHLIGHT = "Highlight"
PROPORTION = "Proportion"

SYNTHETIC = "Synthetic"
AGGREGATE = "Aggregate"

AccuracyModeName = Literal[
    "prioritize_long_combinations", "prioritize_short_combinations", "balanced"
]
FabricationModeName = Literal["balanced", "progressive", "minimized", "uncontrolled"]

# A combination of attribute values, as ``attribute<sep>value`` parts sorted
# lexicographically. pacsynth emits its keys in this order; sorting makes
# comparisons independent of it.
CombinationKey = tuple[str, ...]
Aggregates = dict[CombinationKey, int]


class AnonymizeConfig(BaseModel):
    """Differential privacy parameters and key formatting for one run.

    Defaults match the legacy ``AnonymizeCaseData.anonymize_case_data``
    signature. Modes are plain names so this module never imports pacsynth.
    """

    model_config = ConfigDict(frozen=True)

    epsilon: float = Field(gt=0.0, description="Differential privacy budget")
    reporting_length: int = Field(
        4, ge=1, description="Longest attribute combination to count"
    )
    # pacsynth takes both percentiles as integers; the legacy float annotation
    # only worked because callers passed whole numbers.
    percentile_percentage: int = Field(99, ge=1, le=100)
    percentile_epsilon_proportion: float = Field(0.01, gt=0.0, lt=1.0)
    number_of_records_epsilon_proportion: float = Field(0.005, gt=0.0, lt=1.0)
    weight_selection_percentile: int = Field(95, ge=1, le=100)
    accuracy_mode: AccuracyModeName = "prioritize_long_combinations"
    fabrication_mode: FabricationModeName = "balanced"
    empty_value: str = Field("", description="Text treated as a missing value")
    use_synthetic_counts: bool = True
    aggregate_counts_scale_factor: float = Field(1.0, gt=0.0)
    sensitive_zeros: tuple[str, ...] = Field(
        (),
        description="Columns whose '0' values count as attributes (pacsynth drops them otherwise)",
    )
    attribute_separator: str = Field(";", min_length=1)
    value_separator: str = Field(":", min_length=1)

    @model_validator(mode="after")
    def _check_separators(self) -> AnonymizeConfig:
        if self.attribute_separator == self.value_separator:
            msg = "attribute_separator and value_separator must differ"
            raise ValueError(msg)
        return self


class AttributeValue(BaseModel):
    """One ``attribute = value`` term of a selection."""

    model_config = ConfigDict(frozen=True)

    attribute: str = Field(min_length=1)
    value: str

    def key(self, value_separator: str = ":") -> str:
        return f"{self.attribute}{value_separator}{self.value}"

    @classmethod
    def parse(cls, text: str, value_separator: str = ":") -> AttributeValue:
        attribute, separator, value = text.partition(value_separator)
        if not separator:
            msg = f"{text!r} has no {value_separator!r} between attribute and value"
            raise ValueError(msg)
        return cls(attribute=attribute, value=value)


@dataclass(frozen=True, slots=True)
class RawTable:
    """Header plus rows of text, the form DP engines consume."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        width = len(self.header)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                msg = f"Row {index} has {len(row)} values, header has {width}"
                raise ValueError(msg)

    @classmethod
    def from_lists(cls, data: list[list[str]]) -> RawTable:
        if not data:
            msg = "Raw data needs a header row"
            raise ValueError(msg)
        return cls(tuple(data[0]), tuple(tuple(row) for row in data[1:]))

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> RawTable:
        return cls(
            tuple(frame.columns), tuple(frame.cast(pl.Utf8).fill_null("").rows())
        )

    def as_lists(self) -> list[list[str]]:
        return [list(self.header), *(list(row) for row in self.rows)]

    def to_frame(self) -> pl.DataFrame:
        schema = dict.fromkeys(self.header, pl.Utf8)
        if not self.rows:
            return pl.DataFrame(schema=schema)
        columns = zip(*self.rows, strict=True)
        return pl.DataFrame(
            {
                name: list(values)
                for name, values in zip(self.header, columns, strict=True)
            },
            schema=schema,
        )

    @property
    def height(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class SynthesisOutput:
    """What a differentially private synthesizer hands back."""

    dp_aggregates: dict[str, int]
    protected_record_count: int
    synthetic: RawTable


@dataclass(frozen=True, slots=True)
class SynthesizabilityStatistics:
    """How much combination space the data spans relative to its size."""

    num_cols: int
    overall_att_count: int
    possible_combinations: int
    possible_combinations_per_row: float
    mean_vals_per_record: float
    max_combinations_per_record: float
    excess_combinations_ratio: float


@dataclass(frozen=True, slots=True)
class ErrorRow:
    """Accuracy of one combination length (``length is None`` for overall)."""

    length: int | None
    mean_count: float | None
    mean_error: float | None
    suppressed_pct: float | None
    fabricated_pct: float | None


@dataclass(frozen=True, slots=True)
class ErrorReport:
    """Per-length accuracy of a target aggregate set against the source."""

    rows: tuple[ErrorRow, ...]

    @property
    def overall(self) -> ErrorRow:
        return self.rows[-1]

    @property
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "length": [r.length for r in self.rows],
                "mean_count": [r.mean_count for r in self.rows],
                "mean_error": [r.mean_error for r in self.rows],
                "suppressed_pct": [r.suppressed_pct for r in self.rows],
                "fabricated_pct": [r.fabricated_pct for r in self.rows],
            },
            schema={
                "length": pl.Int64,
                "mean_count": pl.Float64,
                "mean_error": pl.Float64,
                "suppressed_pct": pl.Float64,
                "fabricated_pct": pl.Float64,
            },
        )

    def to_text_frame(self) -> pl.DataFrame:
        """The legacy presentation: strings like ``12.34 +/- 1.20`` and ``3.00 %``."""
        return pl.DataFrame({
            "Length": [
                "Overall" if r.length is None else str(r.length) for r in self.rows
            ],
            "Count +/- Error": [
                f"{_num(r.mean_count):.2f} +/- {_num(r.mean_error):.2f}"
                for r in self.rows
            ],
            "Suppressed %": [f"{_num(r.suppressed_pct):.2f} %" for r in self.rows],
            "Fabricated %": [f"{_num(r.fabricated_pct):.2f} %" for r in self.rows],
        })


def _num(value: float | None) -> float:
    return float("nan") if value is None else value


@dataclass(slots=True)
class AnonymizeResult:
    """Everything produced by one anonymization run.

    The parsed aggregate dictionaries are kept so error reports and queries
    can be recomputed without running the synthesizer again.
    """

    config: AnonymizeConfig
    sensitive: pl.DataFrame
    synthetic: pl.DataFrame
    protected_record_count: int
    delta: float | None
    sensitive_aggregates: Aggregates
    dp_aggregates: Aggregates
    synthetic_aggregates: Aggregates
    aggregate_error_report: ErrorReport | None = None
    synthetic_error_report: ErrorReport | None = None
    raw_dp_aggregates: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def aggregates_frame(self) -> pl.DataFrame:
        from itk_anonymize.aggregate import aggregates_frame

        return aggregates_frame(
            self.dp_aggregates,
            self.protected_record_count,
            self.config.attribute_separator,
        )

    @property
    def synthetic_aggregates_frame(self) -> pl.DataFrame:
        from itk_anonymize.aggregate import aggregates_frame

        return aggregates_frame(
            self.synthetic_aggregates,
            self.protected_record_count,
            self.config.attribute_separator,
        )
