# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from json import dumps, loads

import streamlit as st

import intelligence_toolkit.generate_mock_data.schema_builder as schema_builder


def build_schema_ui(global_schema, last_filename):
    form, preview = st.columns([1, 1])
    processed_filename = last_filename
    with form:
        file = st.file_uploader('Upload schema', type=['json'], key='schema_uploader')
        if file is not None and file.name != last_filename: #sv.loaded_filename.value != file.name:
            processed_filename = file.name
            global_schema.clear()
            jsn = loads(file.read())
            for k, v in jsn.items():
                global_schema[k] = v
        st.markdown('### Edit data schema')
        generate_form_from_json_schema(
            global_schema=global_schema,
            default_schema=schema_builder.create_boilerplate_schema(),
        )
    with preview:
        st.markdown('### Preview')
        schema_tab, object_tab = st.tabs(['JSON schema', 'Sample object'])
        obj = schema_builder.generate_object_from_schema(global_schema)
        with schema_tab:
            st.write(global_schema)
        with object_tab:
            st.write(obj)
        validation = schema_builder.evaluate_object_and_schema(obj, global_schema)
        if validation == schema_builder.ValidationResult.VALID:
            st.success('Schema is valid')
        elif validation == schema_builder.ValidationResult.OBJECT_INVALID:
            st.error('Object is invalid')
        elif validation == schema_builder.ValidationResult.SCHEMA_INVALID:
            st.error('Schema is invalid')
        name = global_schema["title"].replace(" ", "_").lower() + "_[schema].json"
        st.download_button(
            label=f'Download {name}',
            data=dumps(global_schema, indent=2),
            file_name=name,
            mime='application/json'
        )
    return processed_filename

type_options = ['object', 'string', 'number', 'boolean', 'object array', 'string array', 'number array', 'boolean array']

def generate_form_from_json_schema(global_schema, default_schema, field_location=None, nesting=[]):
    # print(f'Generating form for {global_schema}')
    if field_location is None:
        field_location = global_schema
    if type(field_location) != dict:
        return
    for key, value in list(field_location.items()):
        prefix = '.'.join(nesting)
        key_with_prefix = f'{prefix}.{key}' if prefix else key
        if type(value) == dict:
            if 'type' in value and 'description' in value:
                st.divider()
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                with c1:
                    field_type = field_location[key]["type"]
                    if field_type == "array":
                        item_type = field_location[key]["items"]["type"]
                        field_type = f"{item_type} array"
                    new_label = st.text_input(f'Level {len(nesting)} {field_type}', key=f'{key_with_prefix}_label', value=key)
                    if new_label != key:
                        schema_builder.rename_field(global_schema, field_location, nesting, key, new_label)
                        st.rerun()
                with c2:
                    if st.button('Move up', key=f'{key_with_prefix}_move_up', disabled=key == list(field_location.keys())[0]):
                        schema_builder.move_field_up(global_schema, nesting, field_location, key)
                        st.rerun()

                    old_req = key in schema_builder.get_required_list(global_schema, nesting)
                    req = st.checkbox('Required?', key=f'{key_with_prefix}_required', value=old_req)

                    if req != old_req:
                        schema_builder.set_required_field_status(global_schema, nesting, new_label, req)
                        st.rerun()
                with c3:
                    if st.button('Move down', key=f'{key_with_prefix}_move_down', disabled=key == list(field_location.keys())[-1]):
                        schema_builder.move_field_down(global_schema, nesting, field_location, key)
                        st.rerun()

                    show_additional = value["type"] == "object" or (value["type"] == "array" and value["items"]["type"] == "object")
                    if show_additional:
                        add = st.checkbox(
                            "Additional?",
                            key=f"{key_with_prefix}_additional",
                            value=value["additionalProperties"] if "additionalProperties" in value else value["items"]["additionalProperties"]
                        )
                        changed = schema_builder.set_additional_field_status(global_schema, nesting, new_label, add)
                        if changed:
                            st.rerun()

                with c4:
                    if st.button('Delete', use_container_width=True, key=f'{key_with_prefix}_delete'):
                        schema_builder.delete_field(global_schema, nesting, field_location, key)
                        st.rerun()

                    show_enum = value["type"] in ["string", "number"] \
                        or value["type"] == "array" \
                        and value["items"]["type"] in ["string", "number"]
                    if show_enum:
                        con = st.checkbox(
                            "Enum?",
                            key=f"{key_with_prefix}_constrained",
                            value="enum" in value
                            or value["type"] == "array"
                            and "enum" in value["items"]
                        )
                        changed = schema_builder.set_enum_field_status(global_schema, nesting, new_label, con)
                        if changed:
                            st.rerun()
                
                new_description = st.text_input('Field description', key=f'{key_with_prefix}_description', value=value['description'])
                if new_description != value['description']:
                    field_location[new_label]['description'] = new_description
                    st.rerun()

                if 'enum' in value:
                    create_enum_ui(field_location, key, key_with_prefix, value)
                elif value['type'] == 'array' and 'enum' in value['items']:
                    create_enum_ui(field_location, key, key_with_prefix, value['items'])
                else:
                    pass
                    # Open AI structured outputs do not currently support these fields
                    # https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
                    # if value['type'] == 'number':
                    #     create_number_ui(field_location[key], key_with_prefix, value)
                    # elif value['type'] == "string":
                    #     create_string_ui(field_location[key], key_with_prefix, value)
                    # elif value['type'] == 'array':
                    #     if value['items']['type'] == 'number':
                    #         create_number_ui(field_location[key]['items'], key_with_prefix, value['items'])
                    #     elif value['items']['type'] == 'string':
                    #         create_string_ui(field_location[key]['items'], key_with_prefix, value['items'])
                if value['type'] == 'object':
                    generate_form_from_json_schema(global_schema, default_schema, value['properties'], nesting+[key])
                elif value['type'] == 'array':
                    if value['items']['type'] == 'object':
                        generate_form_from_json_schema(global_schema, default_schema, value['items']['properties'], nesting+[key])
            elif key == 'properties' and len(nesting)==0:
                generate_form_from_json_schema(global_schema, default_schema, value, nesting)
                st.divider()
                if st.button('Clear schema', key='clear_schema', use_container_width=True):
                    global_schema.clear()
                    for k, v in default_schema.items():
                        global_schema[k] = v
                    st.rerun()
                return
        else:
            if key != 'type':
                new_value = st.text_input(f'{key}', key=f'{key_with_prefix}_label', value=value)
                if new_value != value:
                    field_location[key] = new_value
                    st.rerun()
    st.divider()
    edit_schema_ui(global_schema, nesting, field_location)

def create_number_ui(number_field, key_with_prefix, value):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        min_value = st.number_input('Min', key=f'{key_with_prefix}_min', value=value.get('minimum', None))
        if min_value != value.get('minimum', None):
            schema_builder.set_number_minimum(number_field, min_value, exclusive=False)
            st.rerun()
    with c2:
        max_value = st.number_input('Max', key=f'{key_with_prefix}_max', value=value.get('maximum', None))
        if max_value != value.get('maximum', None):
            schema_builder.set_number_maximum(number_field, max_value, exclusive=False)
            st.rerun()
    with c3:
        exclusive_min = st.number_input('Exclusive min', key=f'{key_with_prefix}_exclusive_min', value=value.get('exclusiveMinimum', None))
        if exclusive_min != value.get('exclusiveMinimum', None):
            schema_builder.set_number_minimum(number_field, exclusive_min, exclusive=True)
            st.rerun()
    with c4:
        exclusive_max = st.number_input('Exclusive max', key=f'{key_with_prefix}_exclusive_max', value=value.get('exclusiveMaximum', None))
        if exclusive_max != value.get('exclusiveMaximum', None):
            schema_builder.set_number_maximum(number_field, exclusive_max, exclusive=True)
            st.rerun()
    with c5:
        multiple_of = st.number_input('Multiple of', key=f'{key_with_prefix}_multiple_of', value=value.get('multipleOf', None))
        if multiple_of != value.get('multipleOf', None):
            schema_builder.set_number_multiple_of(number_field, multiple_of)
            st.rerun()
    with c6:
        if st.button('Clear', key=f'{key_with_prefix}_clear', use_container_width=True):
            schema_builder.clear_number_constraints(number_field)
            st.rerun()

def create_string_ui(string_field, key_with_prefix, value):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        min_value = st.number_input('Min length', key=f'{key_with_prefix}_min', value=value.get('minLength', None))
        if min_value != value.get('minLength', None):
            schema_builder.set_string_min_length(string_field, min_value, exclusive=False)
            st.rerun()
    with c2:
        max_value = st.number_input('Max length', key=f'{key_with_prefix}_max', value=value.get('maxLength', None))
        if max_value != value.get('maxLength', None):
            schema_builder.set_string_max_length(string_field, max_value, exclusive=False)
            st.rerun()
    with c3:
        pattern = st.text_input('Pattern', key=f'{key_with_prefix}_pattern', value=value.get('pattern', ''))
        if pattern != value.get('pattern', ''):
            schema_builder.set_string_pattern(string_field, pattern)
            st.rerun()
    with c4:
        selected_index = 0 if 'format' not in value else [v.value for v in schema_builder.StringFormat].index(value['format'])+1
        format = st.selectbox('Format', key=f'{key_with_prefix}_format', options=['']+[v.value for v in schema_builder.StringFormat], index=selected_index)
        if format != value.get('format', ''):
            schema_builder.set_string_format(string_field, schema_builder.StringFormat._value2member_map_[format])
            st.rerun()
    with c5:
        if st.button('Clear', key=f'{key_with_prefix}_clear', use_container_width=True):
            schema_builder.clear_string_constraints(string_field)
            st.rerun()

def create_enum_ui(field_location, key, key_with_prefix, value):
    # Create a text input and delete button in a single row for each enum value
    for i, enum_value in enumerate(value['enum']):
        c1, c2 = st.columns([4, 1])
        with c1:
            new_enum_value = st.text_input(f'Enum value {i}', key=f'{key_with_prefix}_enum_{i}', value=enum_value)
            if new_enum_value != str(enum_value):                
                if value['type'] == 'string':
                    value['enum'][i] = new_enum_value
                    st.rerun()
                elif value['type'] == 'number':
                    if new_enum_value.isnumeric():
                        if '.' in new_enum_value:
                            value['enum'][i] = float(new_enum_value)
                        else:
                            value['enum'][i] = int(new_enum_value)
                        st.rerun()
                    else:
                        st.warning('Enum value must be a number')
                
        with c2:
            if st.button('Delete', key=f'{key_with_prefix}_enum_{i}_delete',
                         disabled=(value['enum'][i]=="" or 'enum' in value and len(value['enum'])==1)) or \
                            ('items' in value and 'enum' in value['items'] and len(value['items']['enum'])==1):
                if 'items' in value:
                    value['items']['enum'].pop(i)
                else:
                    value['enum'].pop(i)
                st.rerun()
    new_enum_value = st.text_input(f'New value', key=f'{key_with_prefix}_new_enum_{"_".join([str(x) for x in value["enum"]])}', value="")
    if new_enum_value != "" and new_enum_value not in value['enum']:
        if value['type'] == 'string':
            value['enum'].append(new_enum_value)
            st.rerun()
        elif value['type'] == 'number':
            if new_enum_value.isnumeric():
                if '.' in new_enum_value:
                    value['enum'].append(float(new_enum_value))
                else:
                    value['enum'].append(int(new_enum_value))
                st.rerun()
            else:
                st.warning('Enum value must be a number')

def edit_schema_ui(global_schema, nesting, field_location):
    key_with_prefix = '.'.join(nesting)
    title = f'Add field to `{nesting[-1]}`' if nesting else 'Add top-level field'
    st.markdown(title)
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    with c1:
        if st.button('str', key=f'{key_with_prefix}_add_string', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'string')
            st.rerun()
    with c2:
        if st.button('num', key=f'{key_with_prefix}_add_number', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'number')
            st.rerun()
    with c3:
        if st.button('bool', key=f'{key_with_prefix}_add_boolean', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'boolean')
            st.rerun()
    with c4:
        if st.button('obj', key=f'{key_with_prefix}_add_object', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'object')
            st.rerun()
    with c5:
        if st.button('str[]', key=f'{key_with_prefix}_add_string_array', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'string array')
            st.rerun()
    with c6:
        if st.button('num[]', key=f'{key_with_prefix}_add_number_array', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'number array')
            st.rerun()
    with c7:
        if st.button('bool[]', key=f'{key_with_prefix}_add_boolean_array', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'boolean array')
            st.rerun()
    with c8:
        if st.button('obj[]', key=f'{key_with_prefix}_add_object_array', use_container_width=True):
            add_type_field(global_schema, nesting, field_location, 'object array')
            st.rerun()


def add_type_field(global_schema, nesting, field_location, to_add, required=True):
    label = ''
    if to_add == 'object':
        label = schema_builder.add_object_field(global_schema, field_location)
    elif to_add == 'string':
        label = schema_builder.add_primitive_field(global_schema, field_location, field_type=schema_builder.PrimitiveFieldType.STRING)
    elif to_add == 'number':
        label = schema_builder.add_primitive_field(global_schema, field_location, field_type=schema_builder.PrimitiveFieldType.NUMBER)
    elif to_add == 'boolean':
        label = schema_builder.add_primitive_field(global_schema, field_location, field_type=schema_builder.PrimitiveFieldType.BOOLEAN)
    elif to_add == 'object array':
        label = schema_builder.add_array_field(global_schema, field_location, item_type=schema_builder.ArrayFieldType.OBJECT)
    elif to_add == 'string array':
        label = schema_builder.add_array_field(global_schema, field_location, item_type=schema_builder.ArrayFieldType.STRING)
    elif to_add == 'number array':
        label = schema_builder.add_array_field(global_schema, field_location, item_type=schema_builder.ArrayFieldType.NUMBER)
    elif to_add == 'boolean array':
        label = schema_builder.add_array_field(global_schema, field_location, item_type=schema_builder.ArrayFieldType.BOOLEAN)
    schema_builder.set_required_field_status(global_schema, nesting, label, required)
