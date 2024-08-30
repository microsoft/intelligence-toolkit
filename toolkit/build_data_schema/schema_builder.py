import jsonschema
import json
from enum import Enum

ValidationResult = Enum('ValidationResult', 'VALID SCHEMA_INVALID OBJECT_INVALID')

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

def create_schema(
        schema_field="http://json-schema.org/draft/2020-12/schema",
        id_field="https://yourdomain.com/example.schema.json",
        title_field="Example Schema",
        description_field="An example schema ready to be edited and populated with fields.",
    ):
    schema = {
        "$schema": schema_field,
        "$id": id_field,
        "title": title_field,
        "description": description_field,
    }
    return schema

def add_object_field(
        schema,
        nesting=[],
        field_label="object",
        field_description="An object field"
    ):
    use_schema = get_subobject(schema, nesting)
    use_field_label = _get_unique_field_label(use_schema, field_label)
    use_schema[use_field_label] = {
        "type": "object",
        "description": field_description,
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    return use_field_label

def add_array_field(
        schema,
        nesting=[],
        field_label="",
        field_description="",
        item_description="",
        item_type: ArrayFieldType=ArrayFieldType.STRING
    ):
    use_schema = get_subobject(schema, nesting)
    if field_label == "":
        field_label = f"{item_type.value}_array"
    if field_description == "":
        field_description = f"An array of {item_type.value}s"
    if item_description == "":
        item_description = f"A {item_type.value} list item" if item_type != ArrayFieldType.OBJECT else "An object list item"
    use_field_label = _get_unique_field_label(use_schema, field_label)
    if item_type == ArrayFieldType.OBJECT:
        use_schema[use_field_label] = {
            "type": "array",
            "description": field_description,
            "items": {
                "type": "object",
                "description": item_description,
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    else:
        use_schema[use_field_label] = {
            "type": "array",
            "description": field_description,
            "items": {
                "type": item_type.value
            }
        }
    return use_field_label

def add_primitive_field(
        schema,
        nesting=[],
        field_label="",
        field_description="",
        field_type: PrimitiveFieldType=PrimitiveFieldType.STRING
    ):
    use_schema = get_subobject(schema, nesting)
    if field_label == "":
        field_label = field_type.value
    if field_description == "":
        field_description = f"A {field_type.value} field"
    use_field_label = _get_unique_field_label(use_schema, field_label)
    use_schema[use_field_label] = {
        "type": field_type.value,
        "description": field_description
    }
    return use_field_label

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
                return []
            elif schema['items']['type'] == 'number':
                return []
            elif schema['items']['type'] == 'boolean':
                return []
            else:
                return [generate_template(schema['items'])]
        elif schema['type'] == 'string':
            return ''
        elif schema['type'] == 'number':
            return 0
        elif schema['type'] == 'boolean':
            return False
        else:
            return None

    return generate_template(json_schema)

def evaluate_object_and_schema(obj, schema):
    try:
        jsonschema.validate(obj, schema)
        return ValidationResult.VALID
    except jsonschema.exceptions.ValidationError:
        return ValidationResult.OBJECT_INVALID
    except jsonschema.exceptions.SchemaError:
        return ValidationResult.SCHEMA_INVALID

def evaluate_schema(schema):
    obj = generate_object_from_schema(schema)
    return evaluate_object_and_schema(obj, schema)

def test():
    print('Creating schema')
    schema = create_schema()
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