# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import pytest
from itk_extract.schema import RecordSchema
from itk_extract_testing import PEOPLE_SCHEMA


def test_array_paths_include_nested_arrays(schema: RecordSchema) -> None:
    assert schema.array_paths == [("people",), ("people", "events")]
    assert schema.array_labels == ["people", "people.events"]


def test_title_defaults_when_missing() -> None:
    assert RecordSchema({"type": "object", "properties": {}}).title == "record_object"
    assert RecordSchema(PEOPLE_SCHEMA).title == "People"


def test_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="object at the root"):
        RecordSchema({"type": "array", "items": {"type": "string"}})


def test_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError, match="Invalid JSON schema"):
        RecordSchema({"type": "object", "properties": {"x": {"type": "nope"}}})


def test_validate_reports_violations(schema: RecordSchema) -> None:
    assert schema.validate({"organisation": "Acme", "people": []}) == []
    violations = schema.validate({"organisation": 5, "people": [{"name": 1}]})
    assert any(v.startswith("organisation:") for v in violations)
    assert any(v.startswith("people/0/name:") for v in violations)


def test_for_openai_marks_every_object_strict(schema: RecordSchema) -> None:
    strict = schema.for_openai()
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["organisation", "founded", "people"]
    person = strict["properties"]["people"]["items"]
    assert person["additionalProperties"] is False
    assert person["required"] == ["name", "role", "contact", "events"]
    assert person["properties"]["contact"]["required"] == ["email"]
    # the original is untouched
    assert "required" not in schema.raw


def test_for_openai_keeps_existing_required_order() -> None:
    raw = {
        "type": "object",
        "required": ["b"],
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    assert RecordSchema(raw).for_openai()["required"] == ["b", "a"]


def test_response_format_shape(schema: RecordSchema) -> None:
    fmt = schema.response_format()
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


def test_raw_is_a_copy(schema: RecordSchema) -> None:
    schema.raw["properties"].clear()
    assert schema.array_paths == [("people",), ("people", "events")]
