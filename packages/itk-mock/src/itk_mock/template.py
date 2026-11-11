# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Template objects for a schema, and validation of schemas and objects."""

from __future__ import annotations

from typing import Any

import jsonschema

from itk_mock.models import ValidationResult


def generate_object_from_schema(schema: dict[str, Any]) -> Any:
    """Build the emptiest object the schema describes.

    Primitives are empty (``""``, ``0`` or the nearest value allowed by the
    number constraints, ``False``) or the first enum value; arrays of
    primitives are empty unless constrained to an enum; arrays of objects hold
    one template object.
    """
    kind = schema.get("type")
    enum = schema.get("enum") or []
    if kind == "object":
        if "properties" not in schema:
            return None
        return {
            name: generate_object_from_schema(value)
            for name, value in schema["properties"].items()
        }
    if kind == "array":
        items = schema.get("items", {})
        item_kind = items.get("type")
        if item_kind in {"string", "number"}:
            item_enum = items.get("enum") or []
            return [item_enum[0]] if item_enum else []
        if item_kind == "boolean":
            return []
        return [generate_object_from_schema(items)]
    if kind == "string":
        return enum[0] if enum else ""
    if kind == "number":
        return enum[0] if enum else _constrained_number(schema)
    if kind == "boolean":
        return enum[0] if enum else False
    return None


def evaluate_object_and_schema(obj: Any, schema: dict[str, Any]) -> ValidationResult:
    """Whether the schema is a valid JSON Schema and the object conforms to it."""
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.exceptions.SchemaError:
        return ValidationResult.SCHEMA_INVALID
    except jsonschema.exceptions.ValidationError:
        return ValidationResult.OBJECT_INVALID
    return ValidationResult.VALID


def evaluate_schema(schema: dict[str, Any]) -> ValidationResult:
    """Validate the schema against its own template object."""
    return evaluate_object_and_schema(generate_object_from_schema(schema), schema)


def _constrained_number(schema: dict[str, Any]) -> Any:
    multiple = schema.get("multipleOf")
    if "minimum" in schema:
        minimum = schema["minimum"]
        if multiple:
            return minimum + multiple - (minimum % multiple)
        return minimum
    if "maximum" in schema:
        maximum = schema["maximum"]
        if multiple:
            return maximum - (maximum % multiple)
        return maximum
    return 0
