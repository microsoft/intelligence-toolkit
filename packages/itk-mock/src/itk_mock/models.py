# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs, outputs, enums and errors of the mock data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

MOCK_TEXT = "mock_text"


class MockDataError(ValueError):
    """The model reply or the schema cannot be used to generate records."""


class SchemaEditError(ValueError):
    """A schema editing operation could not be applied."""


class SchemaPathError(SchemaEditError):
    """The nesting path does not lead to an object (or array of objects)."""


class FieldNotFoundError(SchemaEditError):
    """The named field does not exist at the given nesting."""


class DuplicateFieldError(SchemaEditError):
    """A field with that label already exists at the given nesting."""


class FieldTypeError(SchemaEditError):
    """The operation does not apply to a field of this type."""


class ValidationResult(Enum):
    """Outcome of checking an object against a schema."""

    VALID = "valid"
    SCHEMA_INVALID = "schema_invalid"
    OBJECT_INVALID = "object_invalid"


class StringFormat(Enum):
    """JSON Schema string formats."""

    DATE = "date"
    TIME = "time"
    DATE_TIME = "date-time"
    DURATION = "duration"
    URI = "uri"
    EMAIL = "email"
    IDN_EMAIL = "idn-email"
    HOSTNAME = "hostname"
    IDN_HOSTNAME = "idn-hostname"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    REGEX = "regex"
    UUID = "uuid"
    JSON_POINTER = "json-pointer"
    RELATIVE_JSON_POINTER = "relative-json-pointer"
    IRI = "iri"
    IRI_REFERENCE = "iri-reference"
    URI_REFERENCE = "uri-reference"
    URI_TEMPLATE = "uri-template"
    URI_TEMPLATE_EXPRESSION = "uri-template-expression"
    URI_TEMPLATE_FRAGMENT = "uri-template-fragment"


class FieldType(Enum):
    """Every field type the builder can produce."""

    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class ArrayFieldType(Enum):
    """Item types of an array field; arrays of arrays are not allowed."""

    OBJECT = "object"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class PrimitiveFieldType(Enum):
    """Scalar field types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class MockConfig(BaseModel):
    """Knobs for one record generation run."""

    model_config = ConfigDict(frozen=True)

    num_records_overall: int = Field(
        100, ge=1, description="Total primary records to ask the model for"
    )
    records_per_batch: int = Field(
        10, ge=1, description="Primary records requested in a single model call"
    )
    duplicate_records_per_batch: int = Field(
        0, ge=0, description="Near duplicates of the seed record in each seeded batch"
    )
    related_records_per_batch: int = Field(
        0, ge=0, description="Close relations of the seed record in each seeded batch"
    )
    parallel_batches: int = Field(
        50, ge=1, description="Seeded batches generated concurrently per round"
    )
    temperature: float = Field(0.5, ge=0.0, le=1.0)
    generation_guidance: str = Field(
        "", description="Extra instructions passed to the model"
    )
    seed: int | None = Field(
        None, description="Seed for choosing seed records; None is non-deterministic"
    )

    @model_validator(mode="after")
    def _check_counts(self) -> MockConfig:
        if self.num_records_overall < self.records_per_batch:
            msg = "num_records_overall must be at least records_per_batch"
            raise ValueError(msg)
        targeted = self.duplicate_records_per_batch + self.related_records_per_batch
        if targeted > self.records_per_batch:
            msg = (
                "duplicate_records_per_batch + related_records_per_batch "
                "must not exceed records_per_batch"
            )
            raise ValueError(msg)
        return self


class TextConfig(BaseModel):
    """Knobs for one text generation run."""

    model_config = ConfigDict(frozen=True)

    generation_guidance: str = Field(
        "", description="Extra instructions passed to the model"
    )
    temperature: float = Field(0.5, ge=0.0, le=1.0)
    parallelism: int = Field(10, ge=1, description="Texts generated concurrently")


@dataclass(frozen=True, slots=True)
class MockResult:
    """Everything produced by one record generation run.

    ``generated_objects`` holds every parsed model reply in generation order,
    so the merged object and the frames can be rebuilt without the model, and
    ``frames`` feed text generation later.
    """

    schema: dict[str, Any]
    config: MockConfig
    record_arrays: list[list[str]]
    generated_objects: list[dict[str, Any]]
    json_object: dict[str, Any]
    conflicts: list[str]
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)

    @property
    def primary_record_array(self) -> list[str]:
        return self.record_arrays[0]

    @property
    def primary_frame(self) -> pl.DataFrame:
        return self.frames[".".join(self.primary_record_array)]


@dataclass(frozen=True, slots=True)
class TextResult:
    """Texts generated from records, aligned with ``input_texts``."""

    config: TextConfig
    input_texts: list[str]
    texts: list[str]

    @property
    def frame(self) -> pl.DataFrame:
        return text_frame(self.texts)


def text_frame(texts: list[str]) -> pl.DataFrame:
    """One ``mock_text`` column, one row per generated text."""
    return pl.DataFrame({MOCK_TEXT: texts}, schema={MOCK_TEXT: pl.Utf8})
