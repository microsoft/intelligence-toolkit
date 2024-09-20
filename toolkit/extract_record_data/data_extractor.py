# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import asyncio
import random
import re
from json import dumps, loads
from operator import call

import pandas as pd

import toolkit.AI.utils as utils
import toolkit.extract_record_data.prompts as prompts
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback


async def extract_record_data(
    ai_configuration,
    generation_guidance,
    record_arrays,
    data_schema,
    input_texts,
    df_update_callback,
    callback_batch,
):
    generated_objects = []
    current_object_json = {}
    
    new_objects = await _extract_data_parallel(
        ai_configuration=ai_configuration,
        input_texts=input_texts,
        generation_guidance=generation_guidance,
        data_schema=data_schema,
        callbacks=[callback_batch] if callback_batch is not None else None,
    )

    for new_object in new_objects:
        print(new_object)
        new_object_json = loads(new_object)
        generated_objects.append(new_object_json)
        current_object_json, conflicts = merge_json_objects(current_object_json, new_object_json)
    dfs = {}
    for record_array in record_arrays:
        df = extract_df(current_object_json, record_array)
        dfs[record_array] = df
    if df_update_callback is not None:
        df_update_callback(dfs)
    return current_object_json, generated_objects, dfs


async def _extract_data_parallel(
    ai_configuration,
    input_texts,
    generation_guidance,
    data_schema,
    callbacks: list[ProgressBatchCallback] | None = None,
):
    answer_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "record_object",
            "strict": True,
            "schema": data_schema
        }
    }
    mapped_messages = [utils.prepare_messages(
        prompts.data_extraction_prompt, 
        {
            'input_text': input_text,
            'generation_guidance': generation_guidance,
        }) for input_text in input_texts
    ]

    return await utils.map_generate_text(
        ai_configuration,
        mapped_messages,
        response_format=answer_format,
        callbacks=callbacks,
    )

def extract_df(json_data, record_path):
    print(record_path)
    # Extracts a DataFrame from a JSON object
    return pd.json_normalize(
        data=json_data,
        record_path=record_path.split('.')
    )

def merge_json_objects(json_obj1, json_obj2):
    merged_object = {}
    conflicts = []

    def merge_values(key, value1, value2):
        if isinstance(value1, dict) and isinstance(value2, dict):
            merged_value, sub_conflicts = merge_json_objects(value1, value2)
            if sub_conflicts:
                conflicts.extend([f"{key}.{sub_key}" for sub_key in sub_conflicts])
            return merged_value
        elif isinstance(value1, list) and isinstance(value2, list):
            return value1 + value2
        elif value1 != value2:
            conflicts.append(key)
            return value2
        else:
            return value1

    all_keys = set(json_obj1.keys()).union(set(json_obj2.keys()))

    for key in all_keys:
        if key in json_obj1 and key in json_obj2:
            merged_object[key] = merge_values(key, json_obj1[key], json_obj2[key])
        elif key in json_obj1:
            merged_object[key] = json_obj1[key]
        else:
            merged_object[key] = json_obj2[key]

    return merged_object, conflicts


def extract_array_fields(schema):
    # Extracts any array fields at any level of nesting, and returns a list of lists of field names navigating down the schema
    array_fields = []

    def extract_array_fields_recursive(schema, field_path):
        if isinstance(schema, dict):
            for field_name, field_value in schema.get('properties', {}).items():
                if isinstance(field_value, dict):
                    if field_value.get('type') == 'array':
                        array_fields.append(field_path + [field_name])
                        extract_array_fields_recursive(field_value.get('items', {}), field_path + [field_name])
                    else:
                        extract_array_fields_recursive(field_value, field_path + [field_name])
        elif isinstance(schema, list):
            for item in schema:
                extract_array_fields_recursive(item, field_path)

    extract_array_fields_recursive(schema, [])
    return array_fields
