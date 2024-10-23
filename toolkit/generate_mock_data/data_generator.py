# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import random
from json import loads

import pandas as pd

import toolkit.AI.utils as utils
import toolkit.generate_mock_data.prompts as prompts
import toolkit.generate_mock_data.schema_builder as schema_builder
import toolkit.AI.utils as utils
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback


async def generate_data(
    ai_configuration,
    generation_guidance,
    data_schema,
    num_records_overall,
    records_per_batch,
    duplicate_records_per_batch,
    related_records_per_batch,
    temperature,
    df_update_callback,
    callback_batch,
    parallel_batches=0
):
    if parallel_batches == 0:
        parallel_batches = num_records_overall // records_per_batch
    num_iterations = num_records_overall // (records_per_batch * parallel_batches)
    record_arrays = extract_array_fields(data_schema)
    primary_record_array = record_arrays[0]
    generated_objects = []
    first_object = generate_unseeded_data(
                        ai_configuration=ai_configuration,
                        generation_guidance=generation_guidance,
                        primary_record_array=primary_record_array,
                        total_records=parallel_batches,
                        data_schema=data_schema,
                        temperature=temperature,
                    )
    first_object_json = loads(first_object)
    current_object_json = {}
    dfs = {}
    for i in range(num_iterations):
        if i == 0:
            sample_records = sample_from_record_array(
                first_object_json, primary_record_array, records_per_batch
            )
        else:
            sample_records = sample_from_record_array(
                current_object_json, primary_record_array, parallel_batches
            )
        # Use each as seed for parallel gen
        new_objects = await generate_seeded_data(
            ai_configuration=ai_configuration,
            sample_records=sample_records,
            generation_guidance=generation_guidance,
            primary_record_array=primary_record_array,
            total_records=records_per_batch,
            near_duplicate_records=duplicate_records_per_batch,
            close_relation_records=related_records_per_batch,
            data_schema=data_schema,
            temperature=temperature,
            callbacks=[callback_batch] if callback_batch is not None else None,
        )

        for new_object in new_objects:
            new_object_json = loads(new_object)
            generated_objects.append(new_object_json)
            current_object_json, conflicts = merge_json_objects(current_object_json, new_object_json)
        
        for record_array in record_arrays:
            df = extract_df(current_object_json, record_array)
            dfs[".".join(record_array)] = df
        if df_update_callback is not None:
            df_update_callback(dfs)
    return current_object_json, dfs

def generate_unseeded_data(
        ai_configuration,
        generation_guidance,
        primary_record_array,
        total_records,
        data_schema,
        temperature,
    ):

    answer_messages = utils.prepare_messages(
        prompts.unseeded_data_generation_prompt, 
        {
            'generation_guidance': generation_guidance,
            'primary_record_array': primary_record_array,
            'total_records': total_records,
        }
    )
    answer_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer_object", "strict": True, "schema": data_schema},
    }

    return utils.generate_text(
        ai_configuration,
        answer_messages,
        response_format=answer_format,
        temperature=temperature,
    )


async def generate_seeded_data(
    ai_configuration,
    sample_records,
    generation_guidance,
    primary_record_array,
    total_records,
    near_duplicate_records,
    close_relation_records,
    data_schema,
    temperature,
    callbacks: list[ProgressBatchCallback] | None = None,
):
    answer_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_object",
            "strict": True,
            "schema": data_schema
        }
    }
    mapped_messages = [utils.prepare_messages(
        prompts.seeded_data_generation_prompt, 
        {
            'seed_record': sample_record,
            'generation_guidance': generation_guidance,
            'primary_record_array': primary_record_array,
            'record_targets': '\n'.join([
                'Total records: ' + str(total_records),
                'Near duplicates of seed: ' + str(near_duplicate_records),
                'Close relations of seed: ' + str(close_relation_records),
            ]),
        }) for sample_record in sample_records
    ]

    return await utils.map_generate_text(
        ai_configuration,
        mapped_messages,
        response_format=answer_format,
        temperature=temperature,
        callbacks=callbacks,
    )

def select_random_records(num_records, category_to_count):
    select = sum(category_to_count.values())
    selected = random.sample(range(num_records), select)
    # return category to ids
    category_to_ids = {}
    for category, count in category_to_count.items():
        category_to_ids[category] = selected[:count]
        selected = selected[count:]
    return category_to_ids

def extract_df(json_data, record_path):
    # Extracts a DataFrame from a JSON object
    return pd.json_normalize(
        data=json_data,
        record_path=record_path
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


def extract_array_fields(
    schema: dict
) -> list[list[str]]:
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

def sample_from_record_array(current_object, record_array, k):
    records = schema_builder.get_subobject(current_object, record_array)
    return random.sample(records, k) if len(records) > k else records