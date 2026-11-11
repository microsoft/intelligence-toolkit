# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs and outputs of the case pattern detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np

# Columns of the long attribute-links frame produced by itk_patterns.prepare.
SUBJECT = "subject_id"
PERIOD = "period"
ATTRIBUTE_TYPE = "attribute_type"
ATTRIBUTE_VALUE = "attribute_value"
FULL_ATTRIBUTE = "full_attribute"

LINK_SCHEMA = {
    SUBJECT: pl.Int64,
    PERIOD: pl.Utf8,
    ATTRIBUTE_TYPE: pl.Utf8,
    ATTRIBUTE_VALUE: pl.Utf8,
    FULL_ATTRIBUTE: pl.Utf8,
}

# A legacy input column that is never treated as an attribute.
SUBJECT_ID_COLUMN = "Subject ID"

# Columns of the edge frames behind each period graph.
SOURCE = "source"
TARGET = "target"
WEIGHT = "weight"
EDGE_SCHEMA = {SOURCE: pl.Utf8, TARGET: pl.Utf8, WEIGHT: pl.Float64}

# Columns of the detected pattern frame, kept from the legacy workflow.
PATTERN = "pattern"
LENGTH = "length"
COUNT = "count"
MEAN = "mean"
Z_SCORE = "z_score"
DETECTIONS = "detections"
OVERALL_SCORE = "overall_score"
PATTERN_SCHEMA = {
    PERIOD: pl.Utf8,
    PATTERN: pl.Utf8,
    LENGTH: pl.Int64,
    COUNT: pl.Int64,
    MEAN: pl.Float64,
    Z_SCORE: pl.Float64,
    DETECTIONS: pl.Int64,
    OVERALL_SCORE: pl.Float64,
}

# Converging attribute pairs per period.
NODE1 = "node1"
NODE2 = "node2"
PERIOD_COUNT = "period_count"
MEAN_COUNT = "mean_count"
COUNT_DELTA = "count_delta"
COUNT_FACTOR = "count_factor"
PAIR_SCHEMA = {
    PERIOD: pl.Utf8,
    NODE1: pl.Utf8,
    NODE2: pl.Utf8,
    PERIOD_COUNT: pl.Int64,
    MEAN_COUNT: pl.Float64,
    COUNT_DELTA: pl.Float64,
    COUNT_FACTOR: pl.Float64,
}

TIME_SERIES_SCHEMA = {PERIOD: pl.Utf8, PATTERN: pl.Utf8, COUNT: pl.Int64}
ATTRIBUTE_COUNT_SCHEMA = {ATTRIBUTE_VALUE: pl.Utf8, COUNT: pl.Int64}

# Pseudo-periods used as keys of the per-node position dictionaries.
ALL_PERIODS = "ALL"
PRIOR_PREFIX = "<"

PATTERN_JOIN = " & "

ConvergenceTest = Literal["legacy", "vector"]

# node -> period (or ALL_PERIODS, or PRIOR_PREFIX + period) -> embedding row
Positions = dict[str, dict[str, "np.ndarray"]]
# node -> period (or PRIOR_PREFIX + period) -> cosine distance moved
Shifts = dict[str, dict[str, float]]


class PatternConfig(BaseModel):
    """How the attribute model and its per-period embeddings are built.

    Defaults match the legacy ``detect_case_patterns.config`` module and the
    arguments the Streamlit workflow passed.
    """

    model_config = ConfigDict(frozen=True)

    period_column: str = Field(min_length=1)
    attribute_columns: tuple[str, ...] | None = Field(
        None,
        description="Columns treated as attributes; every other column except "
        "the period column and 'Subject ID' when omitted",
    )
    type_value_separator: str = Field(
        "=", min_length=1, description="Joins attribute type and value in a node name"
    )
    min_edge_weight: float = Field(
        0.001, gt=0.0, lt=1.0, description="Weight of the rarest observed pair"
    )
    missing_edge_prop: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Unobserved pairs weigh this fraction of min_edge_weight",
    )
    laplacian: bool = Field(True, description="Normalise the adjacency symmetrically")
    diagonal: bool = Field(True, description="Add self loops before embedding")
    correlation: bool = Field(True, description="Row-normalise the embedding")
    convergence: ConvergenceTest = Field(
        "legacy",
        description="'legacy' compares the second embedding component only, as "
        "the shipped workflow did; 'vector' compares whole embeddings",
    )


class DetectionConfig(BaseModel):
    """Thresholds for turning converging pairs into patterns."""

    model_config = ConfigDict(frozen=True)

    min_pattern_count: int = Field(
        5, ge=1, description="Records a pattern needs in a period to be detected"
    )
    max_pattern_length: int = Field(
        5, ge=1, description="Most attributes a pattern may have"
    )


@dataclass(frozen=True, slots=True)
class ModelSummary:
    links: int
    cases: int
    attributes: int
    periods: int

    def __str__(self) -> str:
        return (
            f"Attribute model has {self.links} links spanning {self.cases} cases, "
            f"{self.attributes} attributes, and {self.periods} periods."
        )


@dataclass(slots=True)
class PatternResult:
    """Everything produced so far by one detection run.

    Stages append to this. The period graphs and the embedding positions are
    kept so ``detect`` can be re-run with new thresholds without re-embedding.
    """

    config: PatternConfig
    source: pl.DataFrame
    links: pl.DataFrame
    period_graphs: dict[str, nx.Graph]
    node_labels: dict[str, int]
    # embed
    positions: Positions | None = field(default=None, repr=False)
    shifts: Shifts | None = field(default=None, repr=False)
    # detect
    detection: DetectionConfig | None = None
    converging_pairs: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=PAIR_SCHEMA)
    )
    all_pairs: int = 0
    patterns: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=PATTERN_SCHEMA)
    )
    time_series: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=TIME_SERIES_SCHEMA)
    )

    @property
    def periods(self) -> list[str]:
        return sorted(self.period_graphs)

    @property
    def attributes(self) -> list[str]:
        return sorted(self.node_labels)

    @property
    def close_pairs(self) -> int:
        return self.converging_pairs.height

    @property
    def summary(self) -> ModelSummary:
        return ModelSummary(
            links=self.links.height,
            cases=self.links.get_column(SUBJECT).n_unique() if self.links.height else 0,
            attributes=len(self.node_labels),
            periods=len(self.period_graphs),
        )

    @property
    def patterns_display(self) -> pl.DataFrame:
        """The legacy presentation: ``mean`` rounded to 0, scores to 2 places."""
        from itk_patterns.detect import display_patterns

        return display_patterns(self.patterns)

    def pattern_series(self, pattern: str) -> pl.DataFrame:
        return self.time_series.filter(pl.col(PATTERN) == pattern)


def split_pattern(pattern: str) -> list[str]:
    return pattern.split(PATTERN_JOIN)


def join_pattern(attributes: list[str] | tuple[str, ...]) -> str:
    return PATTERN_JOIN.join(attributes)


__all__ = [
    "ALL_PERIODS",
    "ATTRIBUTE_COUNT_SCHEMA",
    "EDGE_SCHEMA",
    "LINK_SCHEMA",
    "PAIR_SCHEMA",
    "PATTERN_SCHEMA",
    "PRIOR_PREFIX",
    "TIME_SERIES_SCHEMA",
    "DetectionConfig",
    "ModelSummary",
    "PatternConfig",
    "PatternResult",
    "Positions",
    "Shifts",
    "join_pattern",
    "split_pattern",
]
