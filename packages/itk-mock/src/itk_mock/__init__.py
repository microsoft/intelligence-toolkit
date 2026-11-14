# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Mock data generation: build a schema, generate records, generate texts."""

from itk_mock.chat import ChatFn, openai_chat
from itk_mock.generator import MockDataGenerator
from itk_mock.models import (
    ArrayFieldType,
    DuplicateFieldError,
    FieldNotFoundError,
    FieldType,
    FieldTypeError,
    MockConfig,
    MockDataError,
    MockResult,
    PrimitiveFieldType,
    SchemaEditError,
    SchemaPathError,
    StringFormat,
    TextConfig,
    TextResult,
    ValidationResult,
)

__all__ = [
    "ArrayFieldType",
    "ChatFn",
    "DuplicateFieldError",
    "FieldNotFoundError",
    "FieldType",
    "FieldTypeError",
    "MockConfig",
    "MockDataError",
    "MockDataGenerator",
    "MockResult",
    "PrimitiveFieldType",
    "SchemaEditError",
    "SchemaPathError",
    "StringFormat",
    "TextConfig",
    "TextResult",
    "ValidationResult",
    "openai_chat",
]
