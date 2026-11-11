# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from typing import Any

import pytest
from itk_mock import (
    ArrayFieldType,
    DuplicateFieldError,
    FieldNotFoundError,
    FieldTypeError,
    PrimitiveFieldType,
    SchemaEditError,
    SchemaPathError,
    StringFormat,
)
from itk_mock import schema as sb

RECORDS = ["records"]


def test_boilerplate_has_one_empty_required_records_array() -> None:
    schema = sb.create_boilerplate_schema(title_field="T", description_field="D")

    assert schema["title"] == "T"
    assert schema["description"] == "D"
    assert schema["required"] == ["records"]
    assert schema["properties"]["records"]["items"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_add_fields_with_explicit_labels_are_required_in_field_order(
    boilerplate: dict[str, Any],
) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "name", "Full name")
    sb.add_primitive_field(
        boilerplate, RECORDS, "age", field_type=PrimitiveFieldType.NUMBER
    )
    sb.add_object_field(boilerplate, RECORDS, "address")
    sb.add_primitive_field(boilerplate, ["records", "address"], "city")
    sb.add_array_field(boilerplate, RECORDS, "orders", item_type=ArrayFieldType.OBJECT)
    sb.add_primitive_field(boilerplate, ["records", "orders"], "sku")
    sb.add_array_field(boilerplate, RECORDS, "tags")

    items = boilerplate["properties"]["records"]["items"]
    assert list(items["properties"]) == ["name", "age", "address", "orders", "tags"]
    assert items["required"] == ["name", "age", "address", "orders", "tags"]
    assert items["properties"]["name"] == {
        "type": "string",
        "description": "Full name",
    }
    assert items["properties"]["address"]["required"] == ["city"]
    assert items["properties"]["orders"]["items"]["required"] == ["sku"]
    assert items["properties"]["tags"]["items"] == {"type": "string"}


def test_add_without_label_numbers_the_default_label(
    boilerplate: dict[str, Any],
) -> None:
    first = sb.add_primitive_field(boilerplate, RECORDS)
    second = sb.add_primitive_field(boilerplate, RECORDS)
    numbers = sb.add_array_field(boilerplate, RECORDS, item_type=ArrayFieldType.NUMBER)
    obj = sb.add_object_field(boilerplate, RECORDS)

    assert (first, second, numbers, obj) == (
        "string_1",
        "string_2",
        "number_array_1",
        "object_1",
    )


def test_add_without_label_never_overwrites_after_a_delete(
    boilerplate: dict[str, Any],
) -> None:
    sb.add_primitive_field(boilerplate, RECORDS)
    sb.add_primitive_field(boilerplate, RECORDS)
    sb.delete_field(boilerplate, RECORDS, "string_1")

    assert sb.add_primitive_field(boilerplate, RECORDS) == "string_1"
    assert sb.add_primitive_field(boilerplate, RECORDS) == "string_3"


def test_add_optional_field_is_not_required(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "nickname", required=False)

    assert sb.get_required_list(boilerplate, RECORDS) == []


def test_add_duplicate_or_empty_label_raises(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "name")

    with pytest.raises(DuplicateFieldError, match="'name' already exists at records"):
        sb.add_primitive_field(boilerplate, RECORDS, "name")
    with pytest.raises(SchemaEditError, match="must not be empty"):
        sb.add_primitive_field(boilerplate, RECORDS, "")


def test_unknown_or_non_object_nesting_raises(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "name")

    with pytest.raises(SchemaPathError, match="No field 'missing'"):
        sb.add_primitive_field(boilerplate, ["missing"], "x")
    with pytest.raises(SchemaPathError, match=r"records\.name is not an object"):
        sb.add_primitive_field(boilerplate, ["records", "name"], "x")


def test_rename_keeps_position_and_required_status(
    boilerplate: dict[str, Any],
) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "a")
    sb.add_primitive_field(boilerplate, RECORDS, "b", required=False)
    sb.add_primitive_field(boilerplate, RECORDS, "c")

    sb.rename_field(boilerplate, RECORDS, "a", "first")
    sb.rename_field(boilerplate, RECORDS, "b", "second")

    assert list(sb.get_properties(boilerplate, RECORDS)) == ["first", "second", "c"]
    assert sb.get_required_list(boilerplate, RECORDS) == ["first", "c"]


def test_rename_missing_or_onto_existing_raises(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "a")
    sb.add_primitive_field(boilerplate, RECORDS, "b")

    with pytest.raises(FieldNotFoundError):
        sb.rename_field(boilerplate, RECORDS, "missing", "x")
    with pytest.raises(DuplicateFieldError):
        sb.rename_field(boilerplate, RECORDS, "a", "b")
    assert list(sb.get_properties(boilerplate, RECORDS)) == ["a", "b"]


def test_delete_removes_field_and_required_entry(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "a")
    sb.add_primitive_field(boilerplate, RECORDS, "b")

    sb.delete_field(boilerplate, RECORDS, "a")

    assert list(sb.get_properties(boilerplate, RECORDS)) == ["b"]
    assert sb.get_required_list(boilerplate, RECORDS) == ["b"]
    with pytest.raises(FieldNotFoundError):
        sb.delete_field(boilerplate, RECORDS, "a")


def test_move_up_and_down_reorder_fields_and_required(
    boilerplate: dict[str, Any],
) -> None:
    for label in ("a", "b", "c"):
        sb.add_primitive_field(boilerplate, RECORDS, label)

    sb.move_field_up(boilerplate, RECORDS, "c")
    assert list(sb.get_properties(boilerplate, RECORDS)) == ["a", "c", "b"]
    assert sb.get_required_list(boilerplate, RECORDS) == ["a", "c", "b"]

    sb.move_field_down(boilerplate, RECORDS, "a")
    assert list(sb.get_properties(boilerplate, RECORDS)) == ["c", "a", "b"]

    sb.move_field_up(boilerplate, RECORDS, "c")
    sb.move_field_down(boilerplate, RECORDS, "b")
    assert list(sb.get_properties(boilerplate, RECORDS)) == ["c", "a", "b"]

    with pytest.raises(FieldNotFoundError):
        sb.move_field_up(boilerplate, RECORDS, "missing")


def test_required_status_toggles_and_rejects_unknown_fields(
    boilerplate: dict[str, Any],
) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "a")
    sb.add_primitive_field(boilerplate, RECORDS, "b")

    sb.set_required_field_status(boilerplate, RECORDS, "a", False)
    assert sb.get_required_list(boilerplate, RECORDS) == ["b"]
    sb.set_required_field_status(boilerplate, RECORDS, "a", True)
    assert sb.get_required_list(boilerplate, RECORDS) == ["a", "b"]

    with pytest.raises(FieldNotFoundError):
        sb.set_required_field_status(boilerplate, RECORDS, "missing", True)


@pytest.mark.parametrize(
    ("field_type", "placeholder"),
    [
        (PrimitiveFieldType.STRING, [""]),
        (PrimitiveFieldType.NUMBER, [0]),
        (PrimitiveFieldType.BOOLEAN, [True, False]),
    ],
)
def test_enum_status_on_primitives(
    boilerplate: dict[str, Any], field_type: PrimitiveFieldType, placeholder: list
) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "f", field_type=field_type)

    assert sb.set_enum_field_status(boilerplate, RECORDS, "f", True) is True
    assert sb.get_field(boilerplate, RECORDS, "f")["enum"] == placeholder
    assert sb.set_enum_field_status(boilerplate, RECORDS, "f", True) is False
    assert sb.set_enum_field_status(boilerplate, RECORDS, "f", False) is True
    assert "enum" not in sb.get_field(boilerplate, RECORDS, "f")
    assert sb.set_enum_field_status(boilerplate, RECORDS, "f", False) is False


def test_enum_status_on_arrays_applies_to_items(boilerplate: dict[str, Any]) -> None:
    sb.add_array_field(boilerplate, RECORDS, "tags")
    sb.add_array_field(boilerplate, RECORDS, "flags", item_type=ArrayFieldType.BOOLEAN)
    sb.add_object_field(boilerplate, RECORDS, "obj")

    assert sb.set_enum_field_status(boilerplate, RECORDS, "tags", True) is True
    assert sb.get_field(boilerplate, RECORDS, "tags")["items"]["enum"] == [""]
    assert sb.set_enum_field_status(boilerplate, RECORDS, "flags", True) is False
    assert sb.set_enum_field_status(boilerplate, RECORDS, "obj", True) is False


def test_set_enum_values(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "status")
    sb.add_array_field(boilerplate, RECORDS, "tags")
    sb.add_object_field(boilerplate, RECORDS, "obj")

    sb.set_enum_values(boilerplate, RECORDS, "status", ["open", "closed"])
    sb.set_enum_values(boilerplate, RECORDS, "tags", ("x", "y"))
    assert sb.get_field(boilerplate, RECORDS, "status")["enum"] == ["open", "closed"]
    assert sb.get_field(boilerplate, RECORDS, "tags")["items"]["enum"] == ["x", "y"]

    sb.set_enum_values(boilerplate, RECORDS, "status", None)
    assert "enum" not in sb.get_field(boilerplate, RECORDS, "status")
    with pytest.raises(FieldTypeError):
        sb.set_enum_values(boilerplate, RECORDS, "obj", ["x"])


def test_additional_status_reports_change(boilerplate: dict[str, Any]) -> None:
    sb.add_object_field(boilerplate, RECORDS, "obj")
    sb.add_primitive_field(boilerplate, RECORDS, "name")

    assert sb.set_additional_field_status(boilerplate, RECORDS, "obj", True) is True
    assert sb.get_field(boilerplate, RECORDS, "obj")["additionalProperties"] is True
    assert sb.set_additional_field_status(boilerplate, RECORDS, "obj", True) is False
    assert sb.set_additional_field_status(boilerplate, [], "records", True) is True
    assert boilerplate["properties"]["records"]["items"]["additionalProperties"]
    assert sb.set_additional_field_status(boilerplate, RECORDS, "name", True) is False


def test_string_constraints_set_and_clear(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "email")

    sb.set_string_min_length(boilerplate, RECORDS, "email", 3)
    sb.set_string_max_length(boilerplate, RECORDS, "email", 50)
    sb.set_string_pattern(boilerplate, RECORDS, "email", ".+@.+")
    sb.set_string_format(boilerplate, RECORDS, "email", StringFormat.EMAIL)
    assert sb.get_field(boilerplate, RECORDS, "email") == {
        "type": "string",
        "description": "",
        "minLength": 3,
        "maxLength": 50,
        "pattern": ".+@.+",
        "format": "email",
    }

    sb.set_string_min_length(boilerplate, RECORDS, "email", None)
    assert "minLength" not in sb.get_field(boilerplate, RECORDS, "email")
    sb.clear_string_constraints(boilerplate, RECORDS, "email")
    assert sb.get_field(boilerplate, RECORDS, "email") == {
        "type": "string",
        "description": "",
    }


def test_number_bounds_replace_the_other_kind(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(
        boilerplate, RECORDS, "age", field_type=PrimitiveFieldType.NUMBER
    )

    sb.set_number_minimum(boilerplate, RECORDS, "age", 0)
    sb.set_number_minimum(boilerplate, RECORDS, "age", -1, exclusive=True)
    sb.set_number_maximum(boilerplate, RECORDS, "age", 120, exclusive=True)
    sb.set_number_maximum(boilerplate, RECORDS, "age", 110)
    sb.set_number_multiple_of(boilerplate, RECORDS, "age", 5)
    field = sb.get_field(boilerplate, RECORDS, "age")
    assert field == {
        "type": "number",
        "description": "",
        "exclusiveMinimum": -1,
        "maximum": 110,
        "multipleOf": 5,
    }

    sb.set_number_maximum(boilerplate, RECORDS, "age", None)
    assert "maximum" not in field
    sb.clear_number_constraints(boilerplate, RECORDS, "age")
    assert field == {"type": "number", "description": ""}


def test_constraints_reject_the_wrong_field_type(boilerplate: dict[str, Any]) -> None:
    sb.add_primitive_field(boilerplate, RECORDS, "name")
    sb.add_array_field(boilerplate, RECORDS, "scores", item_type=ArrayFieldType.NUMBER)

    with pytest.raises(FieldTypeError, match="'string', not 'number'"):
        sb.set_number_minimum(boilerplate, RECORDS, "name", 1)
    with pytest.raises(FieldTypeError):
        sb.set_string_pattern(boilerplate, RECORDS, "scores", "x")
    sb.set_number_minimum(boilerplate, RECORDS, "scores", 1)
    assert sb.get_field(boilerplate, RECORDS, "scores")["items"]["minimum"] == 1


def test_normalize_for_openai_requires_everything_and_closes_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                    "required": ["name"],
                },
            }
        },
    }

    normalized = sb.normalize_schema_for_openai(schema)

    assert normalized is schema
    assert schema["required"] == ["records"]
    assert schema["additionalProperties"] is False
    items = schema["properties"]["records"]["items"]
    assert items["required"] == ["name", "address"]
    assert items["additionalProperties"] is False
    assert items["properties"]["address"]["required"] == ["city"]
    assert items["properties"]["address"]["additionalProperties"] is False


def test_extract_array_fields_lists_outer_arrays_first(
    people_schema: dict[str, Any],
) -> None:
    assert sb.extract_array_fields(people_schema) == [
        ["records"],
        ["records", "tags"],
        ["records", "orders"],
    ]
    assert sb.extract_array_fields({"type": "object", "properties": {}}) == []


def test_extract_array_fields_follows_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "wrapper": {
                "type": "object",
                "properties": {"rows": {"type": "array", "items": {"type": "string"}}},
            }
        },
    }

    assert sb.extract_array_fields(schema) == [["wrapper", "rows"]]
