# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import json
from enum import Enum

import jsonschema
import pandas as pd

ValidationResult = Enum('ValidationResult', 'VALID SCHEMA_INVALID OBJECT_INVALID')

class StringFormat(Enum):
    DATE = 'date'
    TIME = 'time'
    DATE_TIME = 'date-time'
    DURATION = 'duration'
    URI = 'uri'
    EMAIL = 'email'
    IDN_EMAIL = 'idn-email'
    HOSTNAME = 'hostname'
    IDN_HOSTNAME = 'idn-hostname'
    IPV4 = 'ipv4'
    IPV6 = 'ipv6'
    REGEX = 'regex'
    UUID = 'uuid'
    JSON_POINTER = 'json-pointer'
    RELATIVE_JSON_POINTER = 'relative-json-pointer'
    IRI = 'iri'
    IRI_REFERENCE = 'iri-reference'
    URI_REFERENCE = 'uri-reference'
    URI_TEMPLATE = 'uri-template'
    URI_TEMPLATE_EXPRESSION = 'uri-template-expression'
    URI_TEMPLATE_FRAGMENT = 'uri-template-fragment'

class FieldType(Enum):
    OBJECT = 'object'
    ARRAY = 'array'
    STRING = 'string'
    NUMBER = 'number'
    BOOLEAN = 'boolean'

class ArrayFieldType(Enum): # Disallow arrays of arrays
    OBJECT = 'object'
    STRING = 'string'
    NUMBER = 'number'
    BOOLEAN = 'boolean'

class PrimitiveFieldType(Enum):
    STRING = 'string'
    NUMBER = 'number'
    BOOLEAN = 'boolean'

def _get_field_label_number(schema, field_label):
    '''
    Returns the number of times the field_label appears in the schema.
    '''
    count = 0
    for key, value in schema.items():
        root_label = '_'.join(key.split('_')[:-1])
        if root_label == field_label:
            count += 1
        if isinstance(value, dict):
            count += _get_field_label_number(value, field_label)
    return count

def _get_unique_field_label(schema, field_label):
    if field_label != "":
        new_suffix = _get_field_label_number(schema, field_label) + 1
        field_label = field_label + '_' + str(new_suffix)
    return field_label

def get_subobject(json_obj, field_labels):
    current_obj = json_obj
    for label in field_labels:
        if label in current_obj:
            current_obj = current_obj[label]
        elif 'properties' in current_obj and label in current_obj['properties']:
            current_obj = current_obj['properties'][label]
        elif 'items' in current_obj and 'properties' in current_obj['items'] and label in current_obj['items']['properties']:
            current_obj = current_obj['items']['properties'][label]
    if 'items' in current_obj:
        current_obj = current_obj['items']['properties']
    elif 'properties' in current_obj:
        current_obj = current_obj['properties']
    return current_obj

def get_required_list(json_obj, field_labels):
    current_obj = json_obj
    for label in field_labels:
        if label in current_obj:
            current_obj = current_obj[label]
        elif 'properties' in current_obj and label in current_obj['properties']:
            current_obj = current_obj['properties'][label]
        elif 'items' in current_obj and 'properties' in current_obj['items'] and label in current_obj['items']['properties']:
            current_obj = current_obj['items']['properties'][label]
    if 'items' in current_obj:
        current_obj = current_obj['items']
    return current_obj['required']

def create_boilerplate_schema(
        schema_field="http://json-schema.org/draft/2020-12/schema",
        title_field="Example Schema",
        description_field="An example schema ready to be edited and populated with fields.",
    ):
    schema = {
        "$schema": schema_field,
        "title": title_field,
        "description": description_field,
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "description": "An array of records",
                "items": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        },
        "required": ["records"],
        "additionalProperties": False
    }
    return schema

def add_object_field(
        global_schema,
        field_location,
        field_label="object",
        field_description=""
    ):
    # if field_description == "":
    #     field_description = f"An object field"
    use_field_label = _get_unique_field_label(global_schema, field_label)
    field_location[use_field_label] = {
        "type": "object",
        "description": field_description,
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    return use_field_label

def add_array_field(
        global_schema,
        field_location,
        field_label="",
        field_description="",
        item_type: ArrayFieldType=ArrayFieldType.STRING
    ):
    if field_label == "":
        field_label = f"{item_type.value}_array"
    # if field_description == "":
    #     field_description = f"An array of {item_type.value}s"
    use_field_label = _get_unique_field_label(global_schema, field_label)
    if item_type == ArrayFieldType.OBJECT:
        field_location[use_field_label] = {
            "type": "array",
            "description": field_description,
            "items": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    else:
        field_location[use_field_label] = {
            "type": "array",
            "description": field_description,
            "items": {
                "type": item_type.value
            }
        }
    return use_field_label

def add_primitive_field(
        global_schema,
        field_location,
        field_label="",
        field_description="",
        field_type: PrimitiveFieldType=PrimitiveFieldType.STRING
    ):
    if field_label == "":
        field_label = field_type.value
    # if field_description == "":
    #     field_description = f"A {field_type.value} field"
    use_field_label = _get_unique_field_label(global_schema, field_label)
    field_location[use_field_label] = {
        "type": field_type.value,
        "description": field_description
    }
    return use_field_label

def set_string_min_length(string_field, min_length):
    if min_length is None:
        string_field.pop('minLength', None)
    else:
        string_field['minLength'] = min_length

def set_string_max_length(string_field, max_length):
    if max_length is None:
        string_field.pop('maxLength', None)
    else:
        string_field['maxLength'] = max_length

def set_string_pattern(string_field, pattern):
    if pattern is None:
        string_field.pop('pattern', None)
    else:
        string_field['pattern'] = pattern

def set_string_format(string_field, string_format: StringFormat | None):
    if string_format is None:
        string_field.pop('format', None)
    else:
        string_field['format'] = string_format.value

def clear_string_constraints(string_field):
    string_field.pop('minLength', None)
    string_field.pop('maxLength', None)
    string_field.pop('pattern', None)
    string_field.pop('format', None)

def set_number_minimum(number_field, minimum, exclusive):
    if minimum is None:
        if exclusive:
            number_field.pop('exclusiveMinimum', None)
        else:
            number_field.pop('minimum', None)
    else:
        if exclusive:
            number_field['exclusiveMinimum'] = minimum
            number_field.pop('minimum', None)
        else:
            number_field['minimum'] = minimum
            number_field.pop('exclusiveMinimum', None)

def set_number_maximum(number_field, maximum, exclusive):
    if maximum is None:
        if exclusive:
            number_field.pop('exclusiveMaximum', None)
        else:
            number_field.pop('maximum', None)
    else:
        if exclusive:
            number_field['exclusiveMaximum'] = maximum
            number_field.pop('maximum', None)
        else:
            number_field['maximum'] = maximum
            number_field.pop('exclusiveMaximum', None)

def set_number_multiple_of(number_field, multiple_of):
    if multiple_of is None:
        number_field.pop('multipleOf', None)
    else:
        number_field['multipleOf'] = multiple_of

def clear_number_constraints(number_field):
    number_field.pop('minimum', None)
    number_field.pop('maximum', None)
    number_field.pop('exclusiveMinimum', None)
    number_field.pop('exclusiveMaximum', None)
    number_field.pop('multipleOf', None)

def rename_field(global_schema, field_location, nesting, old_label, new_label):
    set_required_field_status(global_schema, nesting, old_label, False)
    key_order = list(field_location.keys())
    # Ensures key order is stable
    for key in key_order:
        if key == old_label:
            field_location[new_label] = field_location.pop(key)
        else:
            field_location[key] = field_location.pop(key)
    # Ensures required order matches field order
    set_required_field_status(global_schema, nesting, new_label, True)

def delete_field(global_schema, nesting, field_location, key):
    field_location.pop(key)
    set_required_field_status(global_schema, nesting, key, False)

def move_field_up(global_schema, nesting, field_location, label):
    key_order = list(field_location.keys())
    key_index = key_order.index(label)
    if key_index > 0:
        key_order[key_index - 1], key_order[key_index] = key_order[key_index], key_order[key_index - 1]
        # Ensures key order is stable
        for ix, key in enumerate(key_order[key_index - 1:]):
            field_location[key] = field_location.pop(key)
    reqs = get_required_list(global_schema, nesting)
    reqs.sort(key=lambda x : key_order.index(x))

def move_field_down(global_schema, nesting, field_location, label):
    key_order = list(field_location.keys())
    key_index = key_order.index(label)
    # Move the field down by one position
    if key_index < len(key_order) - 1:
        key_order[key_index + 1], key_order[key_index] = key_order[key_index], key_order[key_index + 1]
        # Ensures key order is stable
        for ix, key in enumerate(key_order[key_index:]):
            field_location[key] = field_location.pop(key)
    reqs = get_required_list(global_schema, nesting)
    reqs.sort(key=lambda x : key_order.index(x))

def set_required_field_status(schema, nesting, field_label, required):
    reqs = get_required_list(schema, nesting)
    if required and field_label not in reqs:
        reqs.append(field_label)
    elif not required and field_label in reqs:
        reqs.remove(field_label)
    obj = get_subobject(schema, nesting)
    key_order = list(obj.keys())
    reqs.sort(key=lambda x : key_order.index(x))


def set_enum_field_status(schema, nesting, field_label, constrained):
    obj = get_subobject(schema, nesting)
    typ = obj[field_label]['type']
    changed = False
    if typ != 'array':
        if constrained and 'enum' not in obj[field_label]:
            changed = True
            if typ == 'string':
                obj[field_label]['enum'] = [""]
            elif typ == 'number':
                obj[field_label]['enum'] = [0]
            elif typ == 'boolean':
                obj[field_label]['enum'] = [True, False]
            else:
                changed = False
        elif not constrained and 'enum' in obj[field_label]:
            obj[field_label].pop('enum')
            changed = True
    else:
        if constrained and 'enum' not in obj[field_label]['items']:
            changed = True
            item_typ = obj[field_label]['items']['type']
            if item_typ == 'string':
                obj[field_label]['items']['enum'] = [""]
            elif item_typ == 'number':
                obj[field_label]['items']['enum'] = [0]
            else:
                changed = False
        elif not constrained and 'enum' in obj[field_label]['items']:
            obj[field_label]['items'].pop('enum')
            changed = True
    return changed

def set_additional_field_status(schema, nesting, field_label, additional):
    obj = get_subobject(schema, nesting)
    typ = obj[field_label]['type']
    changed = False
    if typ == 'object':
        obj[field_label]['additionalProperties'] = additional
    elif typ == 'array' and obj[field_label]['items']['type'] == 'object':
        obj[field_label]['items']['additionalProperties'] = additional
    return changed

def generate_object_from_schema(json_schema):
    '''
    The json_schema is a JSON Schema in which values are described as follows:
    "value": {
        "description": "Description of the value",
        "type": "<type of the value>"
    }
    The type can be "string", "number", "boolean", "array", or "object".
    The generated template contains empty/null values for primitives, empty arrays for arrays of primitives, and empty objects as the sole elements of arrays of objects.
'''
    def generate_template(schema):
        if 'type' not in schema:
            return None
        if schema['type'] == 'object':
            if 'properties' not in schema:
                return None
            return {k: generate_template(v) for k, v in schema['properties'].items()}
        elif schema['type'] == 'array':
            if schema['items']['type'] == 'string':
                if 'enum' in schema['items'] and len(schema['items'] ['enum']) > 0:
                    return [schema['items'] ['enum'][0]]
                else:
                    return []
            elif schema['items']['type'] == 'number':
                if 'enum' in schema['items']  and len(schema['items'] ['enum']) > 0:
                    return [schema['items'] ['enum'][0]]
                else:
                    return []
            elif schema['items']['type'] == 'boolean':
                return []
            else:
                return [generate_template(schema['items'])]
        elif schema['type'] == 'string':
            if 'enum' in schema and len(schema['enum']) > 0:
                return schema['enum'][0]
            else:
                return ''
        elif schema['type'] == 'number':
            if 'enum' in schema and len(schema['enum']) > 0:
                return schema['enum'][0]
            else:
                return _get_constrained_value(schema)
        elif schema['type'] == 'boolean':
            if 'enum' in schema and len(schema['enum']) > 0:
                return schema['enum'][0]
            else:
                return False
        else:
            return None

    return generate_template(json_schema)

def convert_to_dataframe(json_obj):
    df = pd.json_normalize(json_obj)
    return df

def _get_constrained_value(schema):
    if 'minimum' in schema:
        if 'multipleOf' in schema:
            min = schema['minimum']
            mult = schema['multipleOf']
            # find the smallest multiple of mult that is greater than or equal to min
            return min + mult - (min % mult)
        else:
            return schema['minimum']
    elif 'maximum' in schema:
        if 'multipleOf' in schema:
            max = schema['maximum']
            mult = schema['multipleOf']
            # find the largest multiple of mult that is less than or equal to max
            return max - (max % mult)
        else:
            return schema['maximum']
    else:
        return 0

def evaluate_object_and_schema(obj, schema):
    try:
        jsonschema.validate(obj, schema)
        return ValidationResult.VALID
    except jsonschema.exceptions.ValidationError:
        #check if it's is invalid because there's an empty field
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not value or (isinstance(value, str) and value.strip() != ''):
                    return ValidationResult.OBJECT_INVALID
    except jsonschema.exceptions.SchemaError:
        return ValidationResult.SCHEMA_INVALID

def evaluate_schema(schema):
    obj = generate_object_from_schema(schema)
    return evaluate_object_and_schema(obj, schema)

def test():
    print('Creating schema')
    schema = create_boilerplate_schema()
    print(evaluate_schema(schema))

    print('Adding first string field')
    add_primitive_field(
        schema=schema,
        field_type=PrimitiveFieldType.STRING
    )
    print(evaluate_schema(schema))

    print('Adding second string field')
    add_primitive_field(
        schema=schema,
        field_type=PrimitiveFieldType.STRING
    )
    print(evaluate_schema(schema))

    print('Adding nested object field')
    obj_label = add_object_field(
        schema=schema
    )
    print(evaluate_schema(schema))

    print('Adding nested string field')
    add_primitive_field(
        schema=schema,
        nesting=[obj_label],
        field_type=PrimitiveFieldType.STRING
    )
    print(evaluate_schema(schema))

    print('Adding nested string field')
    add_primitive_field(
        schema=schema,
        nesting=[obj_label],
        field_type=PrimitiveFieldType.STRING
    )
    print(evaluate_schema(schema))

    print('Adding nested number field')
    add_primitive_field(
        schema=schema,
        nesting=[obj_label],
        field_type=PrimitiveFieldType.NUMBER
    )
    print(evaluate_schema(schema))

    print('Adding nested boolean field')
    add_primitive_field(
        schema=schema,
        nesting=[obj_label],
        field_type=PrimitiveFieldType.BOOLEAN
    )
    print(evaluate_schema(schema))

    print('Adding nested string array')
    add_array_field(
        schema=schema,
        nesting=[obj_label],
        item_type=PrimitiveFieldType.STRING
    )
    print(evaluate_schema(schema))

    print('Adding nested object array')
    arr_label = add_array_field(
        schema=schema,
        nesting=[obj_label],
        item_type=ArrayFieldType.OBJECT
    )
    print(evaluate_schema(schema))

    print('Adding boolean to objects of nested array')
    add_primitive_field(
        schema=schema,
        nesting=[obj_label, arr_label],
        field_type=PrimitiveFieldType.BOOLEAN
    )
    print(evaluate_schema(schema))

    print('Final schema')
    print(json.dumps(schema, indent=2))

def main():
    test()

if __name__ == "__main__":
    main()