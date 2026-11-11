# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from typing import Any

import pytest
from itk_mock import ValidationResult
from itk_mock.template import (
    evaluate_object_and_schema,
    evaluate_schema,
    generate_object_from_schema,
)


def test_template_for_nested_schema(people_schema: dict[str, Any]) -> None:
    assert generate_object_from_schema(people_schema) == {
        "dataset": "",
        "records": [
            {
                "name": "",
                "age": 0,
                "address": {"city": ""},
                "tags": [],
                "orders": [{"sku": ""}],
            }
        ],
    }


def test_template_uses_first_enum_value() -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["open", "closed"]},
            "level": {"type": "number", "enum": [3, 5]},
            "flag": {"type": "boolean", "enum": [True, False]},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["x"]}},
            "scores": {"type": "array", "items": {"type": "number", "enum": [7]}},
            "flags": {"type": "array", "items": {"type": "boolean"}},
        },
    }

    assert generate_object_from_schema(schema) == {
        "status": "open",
        "level": 3,
        "flag": True,
        "tags": ["x"],
        "scores": [7],
        "flags": [],
    }


@pytest.mark.parametrize(
    ("constraints", "expected"),
    [
        ({}, 0),
        ({"minimum": 18}, 18),
        ({"maximum": 65}, 65),
        ({"minimum": 18, "multipleOf": 5}, 20),
        ({"maximum": 18, "multipleOf": 5}, 15),
        ({"minimum": 18, "maximum": 65}, 18),
    ],
)
def test_template_number_respects_constraints(
    constraints: dict[str, Any], expected: float
) -> None:
    assert generate_object_from_schema({"type": "number", **constraints}) == expected


def test_template_of_untyped_or_propertyless_schema_is_none() -> None:
    assert generate_object_from_schema({}) is None
    assert generate_object_from_schema({"type": "object"}) is None
    assert generate_object_from_schema({"type": "null"}) is None


def test_evaluate_valid_schema_and_template(people_schema: dict[str, Any]) -> None:
    assert evaluate_schema(people_schema) == ValidationResult.VALID


def test_evaluate_object_that_violates_schema(people_schema: dict[str, Any]) -> None:
    obj = {"dataset": "x", "records": [{"name": 3}]}

    assert (
        evaluate_object_and_schema(obj, people_schema)
        == ValidationResult.OBJECT_INVALID
    )


def test_evaluate_template_that_violates_its_own_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 2}},
    }

    assert evaluate_schema(schema) == ValidationResult.OBJECT_INVALID


def test_evaluate_invalid_schema() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "nonsense"}}}

    assert evaluate_schema(schema) == ValidationResult.SCHEMA_INVALID
