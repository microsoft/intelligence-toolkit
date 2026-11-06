# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs and outputs of the matching pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np

RECORD_ID = "record_id"
DATASET = "dataset"
ENTITY_ID = "entity_id"
NAME = "name"
RESERVED_COLUMNS: tuple[str, ...] = (RECORD_ID, DATASET, ENTITY_ID, NAME)

GROUP_ID = "group_id"
GROUP_SIZE = "group_size"
NAME_SIMILARITY = "name_similarity"

NameScorerName = Literal["trigram", "fuzzy"]


class Dataset(BaseModel):
    """One input table and how its columns map onto the shared data model.

    ``attributes`` maps a source column to the canonical label it should carry
    in the unified frame, so columns named differently across datasets (for
    example ``street_address`` and ``address``) can both be mapped to
    ``address``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str = Field(min_length=1)
    frame: pl.DataFrame
    name_column: str
    id_column: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_columns(self) -> Dataset:
        columns = set(self.frame.columns)
        wanted = {self.name_column, *self.attributes}
        if self.id_column is not None:
            wanted.add(self.id_column)
        missing = sorted(wanted - columns)
        if missing:
            msg = f"Dataset {self.name!r} has no columns {missing}"
            raise ValueError(msg)
        clashes = sorted(set(self.attributes.values()) & set(RESERVED_COLUMNS))
        if clashes:
            msg = f"Attribute labels {clashes} are reserved"
            raise ValueError(msg)
        if self.name_column in self.attributes:
            msg = f"Name column {self.name_column!r} cannot also be an attribute"
            raise ValueError(msg)
        return self

    @property
    def labels(self) -> set[str]:
        return set(self.attributes.values())


class MatchConfig(BaseModel):
    """Thresholds controlling candidate generation and grouping."""

    model_config = ConfigDict(frozen=True)

    max_distance: float = Field(
        0.05, ge=0.0, le=2.0, description="Max cosine distance for a candidate pair"
    )
    min_name_similarity: float = Field(
        0.0, ge=0.0, le=1.0, description="Min name score for a pair to join a group"
    )
    neighbors: int = Field(
        50, ge=1, description="Nearest neighbours searched per record"
    )
    name_scorer: NameScorerName = "trigram"
    max_rows_per_dataset: int | None = Field(None, ge=1)


@dataclass(frozen=True, slots=True)
class Pair:
    """Two rows of the unified frame that are candidates for the same entity."""

    left: int
    right: int
    distance: float
    name_similarity: float

    def __post_init__(self) -> None:
        if self.left >= self.right:
            msg = "Pair rows must satisfy left < right"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Group:
    """Rows of the unified frame believed to describe one real-world entity."""

    id: int
    rows: tuple[int, ...]
    name_similarity: float

    @property
    def size(self) -> int:
        return len(self.rows)


@dataclass(slots=True)
class MatchResult:
    """Everything produced by one matching run.

    ``embeddings`` is kept so thresholds can be changed without re-embedding.
    """

    records: pl.DataFrame
    pairs: list[Pair]
    groups: list[Group]
    config: MatchConfig
    embeddings: np.ndarray | None = field(default=None, repr=False)

    @property
    def pairs_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "left": [p.left for p in self.pairs],
                "right": [p.right for p in self.pairs],
                "distance": [p.distance for p in self.pairs],
                NAME_SIMILARITY: [p.name_similarity for p in self.pairs],
            },
            schema={
                "left": pl.UInt32,
                "right": pl.UInt32,
                "distance": pl.Float64,
                NAME_SIMILARITY: pl.Float64,
            },
        )

    @property
    def groups_frame(self) -> pl.DataFrame:
        from itk_match.cluster import groups_frame

        return groups_frame(self.records, self.groups)


class GroupEvaluation(BaseModel):
    """The LLM's judgement of one group."""

    group_id: int
    relatedness: int = Field(ge=0, le=10)
    explanation: str = ""


class EvaluationBatch(BaseModel):
    evaluations: list[GroupEvaluation]
