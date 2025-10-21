# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
from intelligence_toolkit.generate_mock_data.schema_builder import (
    StringFormat,
    FieldType,
    ArrayFieldType,
    PrimitiveFieldType,
    ValidationResult,
    create_boilerplate_schema,
    add_object_field,
    add_array_field,
    add_primitive_field,
    get_subobject,
    get_required_list,
    set_string_min_length,
    set_string_max_length,
    set_string_pattern,
    set_string_format,
    clear_string_constraints,
    set_number_minimum,
    set_number_maximum,
    set_number_multiple_of,
    clear_number_constraints,
    rename_field,
    delete_field,
    move_field_up,
    move_field_down,
    set_required_field_status,
    set_enum_field_status,
    set_additional_field_status,
    normalize_schema_for_openai,
    generate_object_from_schema,
    convert_to_dataframe,
    evaluate_object_and_schema,
    evaluate_schema,
)


def test_string_format_enum():
    assert hasattr(StringFormat, "DATE")
    assert hasattr(StringFormat, "EMAIL")
    assert hasattr(StringFormat, "UUID")
    assert StringFormat.DATE.value == "date"
    assert StringFormat.EMAIL.value == "email"


def test_field_type_enum():
    assert hasattr(FieldType, "OBJECT")
    assert hasattr(FieldType, "ARRAY")
    assert hasattr(FieldType, "STRING")
    assert FieldType.STRING.value == "string"


def test_array_field_type_enum():
    assert hasattr(ArrayFieldType, "OBJECT")
    assert hasattr(ArrayFieldType, "STRING")
    assert not hasattr(ArrayFieldType, "ARRAY")  # No nested arrays


def test_primitive_field_type_enum():
    assert hasattr(PrimitiveFieldType, "STRING")
    assert hasattr(PrimitiveFieldType, "NUMBER")
    assert hasattr(PrimitiveFieldType, "BOOLEAN")


def test_validation_result_enum():
    assert hasattr(ValidationResult, "VALID")
    assert hasattr(ValidationResult, "SCHEMA_INVALID")
    assert hasattr(ValidationResult, "OBJECT_INVALID")


def test_create_boilerplate_schema():
    schema = create_boilerplate_schema()
    
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert "title" in schema
    assert "description" in schema
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "records" in schema["properties"]


def test_create_boilerplate_schema_with_custom_values():
    schema = create_boilerplate_schema(
        schema_field="custom_schema",
        title_field="Custom Title",
        description_field="Custom Description",
    )
    
    assert schema["$schema"] == "custom_schema"
    assert schema["title"] == "Custom Title"
    assert schema["description"] == "Custom Description"


def test_add_primitive_field_string():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label = add_primitive_field(
        schema, field_location, "name", "A name field", PrimitiveFieldType.STRING
    )
    
    assert label in field_location
    assert field_location[label]["type"] == "string"
    assert field_location[label]["description"] == "A name field"


def test_add_primitive_field_number():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label = add_primitive_field(
        schema, field_location, "age", "An age field", PrimitiveFieldType.NUMBER
    )
    
    assert label in field_location
    assert field_location[label]["type"] == "number"


def test_add_primitive_field_boolean():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label = add_primitive_field(
        schema, field_location, "active", "Active status", PrimitiveFieldType.BOOLEAN
    )
    
    assert label in field_location
    assert field_location[label]["type"] == "boolean"


def test_add_object_field():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label = add_object_field(schema, field_location, "address", "An address object")
    
    assert label in field_location
    assert field_location[label]["type"] == "object"
    assert "properties" in field_location[label]
    assert field_location[label]["additionalProperties"] == False


def test_add_array_field_string():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label = add_array_field(
        schema, field_location, "tags", "List of tags", ArrayFieldType.STRING
    )
    
    assert label in field_location
    assert field_location[label]["type"] == "array"
    assert field_location[label]["items"]["type"] == "string"


def test_add_array_field_object():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label = add_array_field(
        schema, field_location, "contacts", "List of contacts", ArrayFieldType.OBJECT
    )
    
    assert label in field_location
    assert field_location[label]["type"] == "array"
    assert field_location[label]["items"]["type"] == "object"
    assert "properties" in field_location[label]["items"]


def test_get_subobject_root():
    schema = create_boilerplate_schema()
    
    result = get_subobject(schema, [])
    
    assert result == schema["properties"]


def test_get_subobject_nested():
    schema = create_boilerplate_schema()
    
    result = get_subobject(schema, ["records"])
    
    assert "properties" in result or isinstance(result, dict)


def test_get_required_list():
    schema = create_boilerplate_schema()
    
    required = get_required_list(schema, [])
    
    assert isinstance(required, list)
    assert "records" in required


def test_set_string_min_length():
    field = {"type": "string"}
    
    set_string_min_length(field, 5)
    
    assert field["minLength"] == 5


def test_set_string_min_length_remove():
    field = {"type": "string", "minLength": 5}
    
    set_string_min_length(field, None)
    
    assert "minLength" not in field


def test_set_string_max_length():
    field = {"type": "string"}
    
    set_string_max_length(field, 100)
    
    assert field["maxLength"] == 100


def test_set_string_pattern():
    field = {"type": "string"}
    
    set_string_pattern(field, "^[A-Z]+$")
    
    assert field["pattern"] == "^[A-Z]+$"


def test_set_string_format():
    field = {"type": "string"}
    
    set_string_format(field, StringFormat.EMAIL)
    
    assert field["format"] == "email"


def test_set_string_format_remove():
    field = {"type": "string", "format": "email"}
    
    set_string_format(field, None)
    
    assert "format" not in field


def test_clear_string_constraints():
    field = {
        "type": "string",
        "minLength": 1,
        "maxLength": 100,
        "pattern": ".*",
        "format": "email",
    }
    
    clear_string_constraints(field)
    
    assert "minLength" not in field
    assert "maxLength" not in field
    assert "pattern" not in field
    assert "format" not in field


def test_set_number_minimum():
    field = {"type": "number"}
    
    set_number_minimum(field, 10, False)
    
    assert field["minimum"] == 10
    assert "exclusiveMinimum" not in field


def test_set_number_minimum_exclusive():
    field = {"type": "number"}
    
    set_number_minimum(field, 10, True)
    
    assert field["exclusiveMinimum"] == 10
    assert "minimum" not in field


def test_set_number_maximum():
    field = {"type": "number"}
    
    set_number_maximum(field, 100, False)
    
    assert field["maximum"] == 100


def test_set_number_multiple_of():
    field = {"type": "number"}
    
    set_number_multiple_of(field, 5)
    
    assert field["multipleOf"] == 5


def test_clear_number_constraints():
    field = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 5,
    }
    
    clear_number_constraints(field)
    
    assert "minimum" not in field
    assert "maximum" not in field
    assert "multipleOf" not in field


def test_rename_field():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "oldname", "", PrimitiveFieldType.STRING)
    
    rename_field(schema, field_location, ["records"], "oldname_1", "newname")
    
    assert "newname" in field_location
    assert "oldname_1" not in field_location


def test_delete_field():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "temp", "", PrimitiveFieldType.STRING)
    
    delete_field(schema, ["records"], field_location, "temp_1")
    
    assert "temp_1" not in field_location


def test_move_field_up():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "first", "", PrimitiveFieldType.STRING)
    add_primitive_field(schema, field_location, "second", "", PrimitiveFieldType.STRING)
    
    keys_before = list(field_location.keys())
    move_field_up(schema, ["records"], field_location, keys_before[1])
    keys_after = list(field_location.keys())
    
    # Second field should now be first
    assert keys_after[0] == keys_before[1]


def test_move_field_down():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "first", "", PrimitiveFieldType.STRING)
    add_primitive_field(schema, field_location, "second", "", PrimitiveFieldType.STRING)
    
    keys_before = list(field_location.keys())
    move_field_down(schema, ["records"], field_location, keys_before[0])
    keys_after = list(field_location.keys())
    
    # First field should now be second
    assert keys_after[1] == keys_before[0]


def test_set_required_field_status_add():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "field", "", PrimitiveFieldType.STRING)
    
    set_required_field_status(schema, ["records"], "field_1", True)
    
    required = get_required_list(schema, ["records"])
    assert "field_1" in required


def test_set_required_field_status_remove():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "field", "", PrimitiveFieldType.STRING)
    set_required_field_status(schema, ["records"], "field_1", True)
    
    set_required_field_status(schema, ["records"], "field_1", False)
    
    required = get_required_list(schema, ["records"])
    assert "field_1" not in required


def test_set_enum_field_status_string():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "status", "", PrimitiveFieldType.STRING)
    
    changed = set_enum_field_status(schema, ["records"], "status_1", True)
    
    assert changed
    assert "enum" in field_location["status_1"]
    assert isinstance(field_location["status_1"]["enum"], list)


def test_set_enum_field_status_remove():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "status", "", PrimitiveFieldType.STRING)
    set_enum_field_status(schema, ["records"], "status_1", True)
    
    changed = set_enum_field_status(schema, ["records"], "status_1", False)
    
    assert changed
    assert "enum" not in field_location["status_1"]


def test_set_additional_field_status():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_object_field(schema, field_location, "obj", "")
    
    set_additional_field_status(schema, ["records"], "obj_1", True)
    
    assert field_location["obj_1"]["additionalProperties"] == True


def test_generate_object_from_schema_simple():
    schema = create_boilerplate_schema()
    
    obj = generate_object_from_schema(schema)
    
    assert isinstance(obj, dict)
    assert "records" in obj


def test_generate_object_from_schema_with_fields():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "name", "", PrimitiveFieldType.STRING)
    add_primitive_field(schema, field_location, "age", "", PrimitiveFieldType.NUMBER)
    
    obj = generate_object_from_schema(schema)
    
    assert isinstance(obj, dict)
    assert "records" in obj
    assert isinstance(obj["records"], list)


def test_generate_object_from_schema_with_enum():
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    add_primitive_field(schema, field_location, "status", "", PrimitiveFieldType.STRING)
    field_location["status_1"]["enum"] = ["active", "inactive"]
    
    obj = generate_object_from_schema(schema)
    
    # Should use first enum value
    assert obj["records"][0]["status_1"] == "active"


def test_convert_to_dataframe():
    json_obj = {"records": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    
    df = convert_to_dataframe(json_obj)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # json_normalize at root level


def test_evaluate_object_and_schema_valid():
    schema = create_boilerplate_schema()
    obj = generate_object_from_schema(schema)
    
    result = evaluate_object_and_schema(obj, schema)
    
    assert result == ValidationResult.VALID


def test_evaluate_schema():
    schema = create_boilerplate_schema()
    
    result = evaluate_schema(schema)
    
    assert result == ValidationResult.VALID


def test_unique_field_labels():
    # Test that adding multiple fields with same label creates unique labels
    schema = create_boilerplate_schema()
    field_location = schema["properties"]["records"]["items"]["properties"]
    
    label1 = add_primitive_field(schema, field_location, "field", "", PrimitiveFieldType.STRING)
    label2 = add_primitive_field(schema, field_location, "field", "", PrimitiveFieldType.STRING)
    
    assert label1 != label2
    assert label1 in field_location
    assert label2 in field_location


def test_normalize_schema_for_openai():
    """Test that imported schemas are normalized for OpenAI structured outputs."""
    # Create a schema without required fields or additionalProperties
    incomplete_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"}
                }
            },
            "hobbies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            }
        }
    }
    
    # Normalize the schema
    normalized = normalize_schema_for_openai(incomplete_schema)
    
    # Check root level
    assert "required" in normalized
    assert "additionalProperties" in normalized
    assert normalized["additionalProperties"] == False
    assert "name" in normalized["required"]
    assert "age" in normalized["required"]
    assert "address" in normalized["required"]
    assert "hobbies" in normalized["required"]
    
    # Check nested object
    address_obj = normalized["properties"]["address"]
    assert "required" in address_obj
    assert "additionalProperties" in address_obj
    assert address_obj["additionalProperties"] == False
    assert "street" in address_obj["required"]
    assert "city" in address_obj["required"]
    
    # Check array items object
    hobby_items = normalized["properties"]["hobbies"]["items"]
    assert "required" in hobby_items
    assert "additionalProperties" in hobby_items
    assert hobby_items["additionalProperties"] == False
    assert "name" in hobby_items["required"]
    assert "category" in hobby_items["required"]


def test_normalize_schema_preserves_existing_required():
    """Test that existing required fields are preserved during normalization."""
    schema_with_partial_required = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {"type": "string"}
        },
        "required": ["name"],  # Only name is currently required
        "additionalProperties": True  # This should be changed to False
    }
    
    normalized = normalize_schema_for_openai(schema_with_partial_required)
    
    # Should preserve existing required field and add new ones
    assert "name" in normalized["required"]
    assert "age" in normalized["required"]
    assert "email" in normalized["required"]
    
    # Should force additionalProperties to False
    assert normalized["additionalProperties"] == False
