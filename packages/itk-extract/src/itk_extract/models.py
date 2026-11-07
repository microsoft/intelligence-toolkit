# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs and outputs of the extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from itk_extract.schema import RecordSchema

SOURCE_ID = "source_id"
CHUNK_ID = "chunk_id"
PROVENANCE_COLUMNS: tuple[str, ...] = (SOURCE_ID, CHUNK_ID)

MergePolicy = Literal["last", "first", "majority", "keep_all"]


class Source(BaseModel):
    """One piece of unstructured text to extract records from."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ExtractConfig(BaseModel):
    """Knobs for one extraction run."""

    model_config = ConfigDict(frozen=True)

    guidance: str = Field("", description="Extra instructions passed to the model")
    chunk_tokens: int | None = Field(
        None,
        ge=1,
        description="Split sources longer than this many tokens; None keeps them whole",
    )
    merge_policy: MergePolicy = Field(
        "last", description="How scalar values that differ across chunks are resolved"
    )
    annotate_provenance: bool = Field(
        True, description="Tag array elements with the source and chunk they came from"
    )
    dedupe_records: bool = Field(
        False,
        description="Drop identical records (ignoring provenance) in output frames",
    )
    validate_output: bool = Field(
        True, description="Check every model reply against the JSON schema"
    )
    fail_fast: bool = Field(
        False, description="Raise on the first unusable reply instead of recording it"
    )
    max_sources: int | None = Field(None, ge=1)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A span of one source's text that is sent to the model in a single call."""

    id: str
    source_id: str
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class Extraction:
    """A model reply that parsed (and, if asked, validated) against the schema."""

    chunk: Chunk
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionError:
    """A model reply that could not be used."""

    chunk: Chunk
    message: str
    raw: str


@dataclass(frozen=True, slots=True)
class Conflict:
    """Differing scalar values for one path across chunks, and the one kept.

    ``values`` holds ``(chunk_id, value)`` pairs in extraction order.
    """

    path: str
    chosen: Any
    values: tuple[tuple[str, Any], ...]


@dataclass(slots=True)
class ExtractResult:
    """Everything produced by one extraction run.

    ``extractions`` are kept so the merge policy can be changed without calling
    the model again (see ``Extractor.remerge``).
    """

    schema: RecordSchema
    config: ExtractConfig
    extractions: list[Extraction]
    errors: list[ExtractionError]
    object: dict[str, Any]
    conflicts: list[Conflict]
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)

    @property
    def conflicts_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "path": [c.path for c in self.conflicts],
                "chosen": [_as_text(c.chosen) for c in self.conflicts],
                CHUNK_ID: [[cid for cid, _ in c.values] for c in self.conflicts],
                "values": [[_as_text(v) for _, v in c.values] for c in self.conflicts],
            },
            schema={
                "path": pl.Utf8,
                "chosen": pl.Utf8,
                CHUNK_ID: pl.List(pl.Utf8),
                "values": pl.List(pl.Utf8),
            },
        )

    @property
    def errors_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                CHUNK_ID: [e.chunk.id for e in self.errors],
                SOURCE_ID: [e.chunk.source_id for e in self.errors],
                "message": [e.message for e in self.errors],
            },
            schema={CHUNK_ID: pl.Utf8, SOURCE_ID: pl.Utf8, "message": pl.Utf8},
        )


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)
