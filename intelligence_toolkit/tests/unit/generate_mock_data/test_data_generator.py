# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from intelligence_toolkit.generate_mock_data.data_generator import (
    extract_array_fields,
    extract_df,
    merge_json_objects,
    select_random_records,
    sample_from_record_array,
)


def test_extract_array_fields_simple():
    schema = {
        "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {}}}
        }
    }
    
    result = extract_array_fields(schema)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert ["items"] in result


def test_extract_array_fields_nested():
    schema = {
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "array",
                        "items": {"type": "object", "properties": {}},
                    }
                },
            }
        }
    }
    
    result = extract_array_fields(schema)
    
    assert ["level1", "level2"] in result


def test_extract_array_fields_multiple_arrays():
    schema = {
        "properties": {
            "array1": {"type": "array", "items": {"type": "string"}},
            "array2": {"type": "array", "items": {"type": "number"}},
        }
    }
    
    result = extract_array_fields(schema)
    
    assert len(result) >= 2
    assert ["array1"] in result
    assert ["array2"] in result


def test_extract_array_fields_no_arrays():
    schema = {
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "number"},
        }
    }
    
    result = extract_array_fields(schema)
    
    assert result == []


def test_extract_df_simple():
    json_data = {"records": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    record_path = ["records"]
    
    df = extract_df(json_data, record_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns
    assert "age" in df.columns


def test_extract_df_empty():
    json_data = {"records": []}
    record_path = ["records"]
    
    df = extract_df(json_data, record_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_merge_json_objects_simple():
    obj1 = {"a": 1, "b": 2}
    obj2 = {"c": 3, "d": 4}
    
    merged, conflicts = merge_json_objects(obj1, obj2)
    
    assert merged == {"a": 1, "b": 2, "c": 3, "d": 4}
    assert conflicts == []


def test_merge_json_objects_with_arrays():
    obj1 = {"items": [1, 2, 3]}
    obj2 = {"items": [4, 5, 6]}
    
    merged, conflicts = merge_json_objects(obj1, obj2)
    
    assert merged["items"] == [1, 2, 3, 4, 5, 6]
    assert conflicts == []


def test_merge_json_objects_with_conflicts():
    obj1 = {"value": 10}
    obj2 = {"value": 20}
    
    merged, conflicts = merge_json_objects(obj1, obj2)
    
    assert merged["value"] == 20  # obj2 wins
    assert "value" in conflicts


def test_merge_json_objects_nested():
    obj1 = {"nested": {"a": 1, "b": 2}}
    obj2 = {"nested": {"c": 3}}
    
    merged, conflicts = merge_json_objects(obj1, obj2)
    
    assert merged["nested"]["a"] == 1
    assert merged["nested"]["c"] == 3


def test_merge_json_objects_nested_arrays():
    obj1 = {"data": {"items": [1, 2]}}
    obj2 = {"data": {"items": [3, 4]}}
    
    merged, conflicts = merge_json_objects(obj1, obj2)
    
    assert merged["data"]["items"] == [1, 2, 3, 4]


def test_select_random_records_single_category():
    category_to_count = {"duplicates": 3}
    
    result = select_random_records(10, category_to_count)
    
    assert "duplicates" in result
    assert len(result["duplicates"]) == 3
    assert all(0 <= idx < 10 for idx in result["duplicates"])


def test_select_random_records_multiple_categories():
    category_to_count = {"duplicates": 2, "relations": 3}
    
    result = select_random_records(20, category_to_count)
    
    assert "duplicates" in result
    assert "relations" in result
    assert len(result["duplicates"]) == 2
    assert len(result["relations"]) == 3


def test_select_random_records_no_overlap():
    category_to_count = {"cat1": 5, "cat2": 5}
    
    result = select_random_records(20, category_to_count)
    
    # Selected IDs should not overlap
    all_ids = result["cat1"] + result["cat2"]
    assert len(all_ids) == len(set(all_ids))


@patch("intelligence_toolkit.generate_mock_data.data_generator.schema_builder.get_subobject")
def test_sample_from_record_array_sufficient_records(mock_get_subobject):
    mock_get_subobject.return_value = [1, 2, 3, 4, 5]
    current_object = {"records": []}
    record_array = ["records"]
    
    result = sample_from_record_array(current_object, record_array, 3)
    
    assert len(result) == 3
    assert all(r in [1, 2, 3, 4, 5] for r in result)


@patch("intelligence_toolkit.generate_mock_data.data_generator.schema_builder.get_subobject")
def test_sample_from_record_array_insufficient_records(mock_get_subobject):
    mock_get_subobject.return_value = [1, 2]
    current_object = {"records": []}
    record_array = ["records"]
    
    result = sample_from_record_array(current_object, record_array, 5)
    
    # Should return all available when k > available
    assert len(result) == 2
    assert result == [1, 2]


def test_extract_df_with_nested_data():
    json_data = {
        "users": [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
    }
    record_path = ["users"]
    
    df = extract_df(json_data, record_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns


def test_merge_json_objects_preserves_both_sides():
    obj1 = {"unique1": "value1", "shared": "old"}
    obj2 = {"unique2": "value2", "shared": "new"}
    
    merged, conflicts = merge_json_objects(obj1, obj2)
    
    assert "unique1" in merged
    assert "unique2" in merged
    assert merged["shared"] == "new"
    assert "shared" in conflicts
