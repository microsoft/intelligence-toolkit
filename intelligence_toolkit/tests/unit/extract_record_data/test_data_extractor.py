# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for extract_record_data data_extractor module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from intelligence_toolkit.extract_record_data.data_extractor import (
    extract_array_fields,
    extract_df,
    extract_record_data,
    merge_json_objects,
)


class TestExtractRecordData:
    @pytest.mark.asyncio
    async def test_extract_record_data_basic(self) -> None:
        """Test basic record data extraction."""
        ai_config = MagicMock()
        input_texts = ["Test text 1", "Test text 2"]
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        with patch(
            "intelligence_toolkit.extract_record_data.data_extractor._extract_data_parallel"
        ) as mock_extract:
            mock_extract.return_value = ['{"name": "John"}', '{"name": "Jane"}']
            
            result_json, result_dfs = await extract_record_data(
                ai_configuration=ai_config,
                generation_guidance="Test guidance",
                record_arrays=[],
                data_schema=schema,
                input_texts=input_texts,
                df_update_callback=None,
                callback_batch=None,
            )
            
            assert result_json == {"name": "Jane"}  # Last one wins in merge
            assert isinstance(result_dfs, dict)
            mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_record_data_with_arrays(self) -> None:
        """Test record data extraction with array fields."""
        ai_config = MagicMock()
        input_texts = ["Test text"]
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {"type": "number"}}}
                }
            }
        }
        
        with patch(
            "intelligence_toolkit.extract_record_data.data_extractor._extract_data_parallel"
        ) as mock_extract:
            mock_extract.return_value = ['{"items": [{"id": 1}]}']
            
            result_json, result_dfs = await extract_record_data(
                ai_configuration=ai_config,
                generation_guidance="",
                record_arrays=[["items"]],
                data_schema=schema,
                input_texts=input_texts,
                df_update_callback=None,
                callback_batch=None,
            )
            
            assert "items" in result_json
            assert "items" in result_dfs

    @pytest.mark.asyncio
    async def test_extract_record_data_with_callback(self) -> None:
        """Test record data extraction with dataframe update callback."""
        ai_config = MagicMock()
        input_texts = ["Test text"]
        schema = {"type": "object"}
        callback = MagicMock()
        
        with patch(
            "intelligence_toolkit.extract_record_data.data_extractor._extract_data_parallel"
        ) as mock_extract:
            mock_extract.return_value = ['{"data": "value"}']
            
            await extract_record_data(
                ai_configuration=ai_config,
                generation_guidance="",
                record_arrays=[],
                data_schema=schema,
                input_texts=input_texts,
                df_update_callback=callback,
                callback_batch=None,
            )
            
            callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_record_data_with_batch_callback(self) -> None:
        """Test record data extraction with batch callback."""
        ai_config = MagicMock()
        input_texts = ["Test text"]
        schema = {"type": "object"}
        batch_callback = MagicMock()
        
        with patch(
            "intelligence_toolkit.extract_record_data.data_extractor._extract_data_parallel"
        ) as mock_extract:
            mock_extract.return_value = ['{"data": "value"}']
            
            await extract_record_data(
                ai_configuration=ai_config,
                generation_guidance="",
                record_arrays=[],
                data_schema=schema,
                input_texts=input_texts,
                df_update_callback=None,
                callback_batch=batch_callback,
            )
            
            call_args = mock_extract.call_args
            assert call_args[1]["callbacks"] == [batch_callback]


class TestExtractDataParallel:
    @pytest.mark.asyncio
    async def test_extract_data_parallel_basic(self) -> None:
        """Test parallel data extraction."""
        from intelligence_toolkit.extract_record_data.data_extractor import _extract_data_parallel
        
        ai_config = MagicMock()
        input_texts = ["Text 1", "Text 2"]
        schema = {"type": "object"}
        
        with patch(
            "intelligence_toolkit.extract_record_data.data_extractor.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.extract_record_data.data_extractor.utils.map_generate_text"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = ['{"result": 1}', '{"result": 2}']
            
            result = await _extract_data_parallel(
                ai_configuration=ai_config,
                input_texts=input_texts,
                generation_guidance="Test guidance",
                data_schema=schema,
                callbacks=None,
            )
            
            assert len(result) == 2
            assert mock_prepare.call_count == 2
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_data_parallel_with_callbacks(self) -> None:
        """Test parallel data extraction with callbacks."""
        from intelligence_toolkit.extract_record_data.data_extractor import _extract_data_parallel
        
        ai_config = MagicMock()
        input_texts = ["Text 1"]
        schema = {"type": "object"}
        callbacks = [MagicMock()]
        
        with patch(
            "intelligence_toolkit.extract_record_data.data_extractor.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.extract_record_data.data_extractor.utils.map_generate_text"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = ['{"result": 1}']
            
            await _extract_data_parallel(
                ai_configuration=ai_config,
                input_texts=input_texts,
                generation_guidance="",
                data_schema=schema,
                callbacks=callbacks,
            )
            
            call_args = mock_generate.call_args
            assert call_args[1]["callbacks"] == callbacks


class TestExtractDf:
    def test_extract_df_basic(self) -> None:
        """Test extracting DataFrame from JSON data."""
        json_data = {
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"}
            ]
        }
        
        result = extract_df(json_data, ["items"])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "name" in result.columns

    def test_extract_df_nested(self) -> None:
        """Test extracting DataFrame from nested JSON data."""
        json_data = {
            "users": [
                {
                    "name": "John",
                    "contacts": [
                        {"type": "email", "value": "john@test.com"}
                    ]
                }
            ]
        }
        
        result = extract_df(json_data, ["users", "contacts"])
        
        assert isinstance(result, pd.DataFrame)
        assert "type" in result.columns
        assert "value" in result.columns

    def test_extract_df_empty(self) -> None:
        """Test extracting DataFrame from empty array."""
        json_data = {"items": []}
        
        result = extract_df(json_data, ["items"])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestMergeJsonObjects:
    def test_merge_json_objects_simple(self) -> None:
        """Test merging simple JSON objects."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"c": 3, "d": 4}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged == {"a": 1, "b": 2, "c": 3, "d": 4}
        assert conflicts == []

    def test_merge_json_objects_overlap_same_values(self) -> None:
        """Test merging objects with overlapping keys and same values."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"b": 2, "c": 3}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged == {"a": 1, "b": 2, "c": 3}
        assert conflicts == []

    def test_merge_json_objects_overlap_different_values(self) -> None:
        """Test merging objects with overlapping keys and different values."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"b": 3, "c": 4}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged["a"] == 1
        assert merged["b"] == 3  # obj2 value wins
        assert merged["c"] == 4
        assert "b" in conflicts

    def test_merge_json_objects_nested(self) -> None:
        """Test merging nested JSON objects."""
        obj1 = {"user": {"name": "John", "age": 30}}
        obj2 = {"user": {"age": 30, "city": "NYC"}}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged["user"]["name"] == "John"
        assert merged["user"]["age"] == 30
        assert merged["user"]["city"] == "NYC"
        assert conflicts == []

    def test_merge_json_objects_nested_conflict(self) -> None:
        """Test merging nested objects with conflicts."""
        obj1 = {"user": {"name": "John", "age": 30}}
        obj2 = {"user": {"name": "Jane", "age": 30}}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged["user"]["name"] == "Jane"  # obj2 wins
        assert "user.name" in conflicts

    def test_merge_json_objects_arrays(self) -> None:
        """Test merging objects with array values."""
        obj1 = {"items": [1, 2, 3]}
        obj2 = {"items": [4, 5, 6]}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged["items"] == [1, 2, 3, 4, 5, 6]
        assert conflicts == []

    def test_merge_json_objects_empty(self) -> None:
        """Test merging with empty objects."""
        obj1 = {"a": 1}
        obj2 = {}
        
        merged, conflicts = merge_json_objects(obj1, obj2)
        
        assert merged == {"a": 1}
        assert conflicts == []

    def test_merge_json_objects_both_empty(self) -> None:
        """Test merging two empty objects."""
        merged, conflicts = merge_json_objects({}, {})
        
        assert merged == {}
        assert conflicts == []


class TestExtractArrayFields:
    def test_extract_array_fields_single(self) -> None:
        """Test extracting a single array field."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        result = extract_array_fields(schema)
        
        assert len(result) == 1
        assert ["items"] in result

    def test_extract_array_fields_nested(self) -> None:
        """Test extracting nested array fields."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "contacts": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        
        result = extract_array_fields(schema)
        
        assert len(result) == 2
        assert ["users"] in result
        assert ["users", "contacts"] in result

    def test_extract_array_fields_multiple(self) -> None:
        """Test extracting multiple array fields at same level."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "users": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        result = extract_array_fields(schema)
        
        assert len(result) == 2
        assert ["items"] in result
        assert ["users"] in result

    def test_extract_array_fields_none(self) -> None:
        """Test extracting when no array fields exist."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        result = extract_array_fields(schema)
        
        assert len(result) == 0

    def test_extract_array_fields_empty_schema(self) -> None:
        """Test extracting from empty schema."""
        result = extract_array_fields({})
        
        assert len(result) == 0

    def test_extract_array_fields_deeply_nested(self) -> None:
        """Test extracting deeply nested array fields."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level2": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "level3": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        result = extract_array_fields(schema)
        
        assert len(result) == 3
        assert ["level1"] in result
        assert ["level1", "level2"] in result
        assert ["level1", "level2", "level3"] in result

    def test_extract_array_fields_schema_as_list(self) -> None:
        """Test extracting array fields when schema contains list structures."""
        # Test the edge case where schema itself might be a list
        # This tests lines 129-131 in data_extractor.py
        schema = [
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        ]
        
        result = extract_array_fields(schema)
        
        assert len(result) == 1
        assert ["items"] in result
