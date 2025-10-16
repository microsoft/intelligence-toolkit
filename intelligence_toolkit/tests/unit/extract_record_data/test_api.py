# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for extract_record_data api module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration
from intelligence_toolkit.extract_record_data.api import ExtractRecordData


class TestExtractRecordData:
    """Base test class for ExtractRecordData."""

    @pytest.fixture
    def api_instance(self) -> ExtractRecordData:
        """Create an ExtractRecordData instance for testing."""
        return ExtractRecordData()

    @pytest.fixture
    def sample_schema(self) -> dict:
        """Create a sample JSON schema for testing."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "emails": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

    @pytest.fixture
    def ai_config(self) -> OpenAIConfiguration:
        """Create a mock AI configuration."""
        config = MagicMock(spec=OpenAIConfiguration)
        return config


class TestInitialization(TestExtractRecordData):
    def test_initialization(self, api_instance) -> None:
        """Test that ExtractRecordData initializes with correct default values."""
        assert api_instance.json_schema == {}
        assert api_instance.record_arrays == []
        assert api_instance.json_object == {}
        assert api_instance.array_dfs == {}


class TestSetSchema(TestExtractRecordData):
    def test_set_schema_basic(self, api_instance, sample_schema) -> None:
        """Test setting a basic schema."""
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_generator.extract_array_fields"
        ) as mock_extract:
            mock_extract.return_value = [["emails"]]
            
            api_instance.set_schema(sample_schema)
            
            assert api_instance.json_schema == sample_schema
            assert api_instance.record_arrays == [["emails"]]
            mock_extract.assert_called_once_with(sample_schema)

    def test_set_schema_empty(self, api_instance) -> None:
        """Test setting an empty schema."""
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_generator.extract_array_fields"
        ) as mock_extract:
            mock_extract.return_value = []
            
            api_instance.set_schema({})
            
            assert api_instance.json_schema == {}
            assert api_instance.record_arrays == []

    def test_set_schema_with_nested_arrays(self, api_instance) -> None:
        """Test setting a schema with nested arrays."""
        nested_schema = {
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
        
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_generator.extract_array_fields"
        ) as mock_extract:
            mock_extract.return_value = [["users"], ["users", "contacts"]]
            
            api_instance.set_schema(nested_schema)
            
            assert api_instance.json_schema == nested_schema
            assert len(api_instance.record_arrays) == 2


class TestSetAIConfiguration(TestExtractRecordData):
    def test_set_ai_configuration(self, api_instance, ai_config) -> None:
        """Test setting AI configuration."""
        api_instance.set_ai_configuration(ai_config)
        
        assert api_instance.ai_configuration == ai_config


class TestExtractRecordDataMethod(TestExtractRecordData):
    @pytest.mark.asyncio
    async def test_extract_record_data_basic(self, api_instance, sample_schema, ai_config) -> None:
        """Test basic record data extraction."""
        api_instance.set_schema(sample_schema)
        api_instance.set_ai_configuration(ai_config)
        
        input_texts = ["John is 30 years old", "Jane is 25 years old"]
        
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_extractor.extract_record_data"
        ) as mock_extract:
            mock_json = {"name": "John", "age": 30}
            mock_dfs = {"emails": MagicMock()}
            mock_extract.return_value = (mock_json, mock_dfs)
            
            await api_instance.extract_record_data(input_texts)
            
            assert api_instance.json_object == mock_json
            assert api_instance.array_dfs == mock_dfs
            mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_record_data_with_guidance(self, api_instance, sample_schema, ai_config) -> None:
        """Test record data extraction with generation guidance."""
        api_instance.set_schema(sample_schema)
        api_instance.set_ai_configuration(ai_config)
        
        input_texts = ["Text about a person"]
        guidance = "Extract only verified information"
        
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_extractor.extract_record_data"
        ) as mock_extract:
            mock_extract.return_value = ({}, {})
            
            await api_instance.extract_record_data(input_texts, generation_guidance=guidance)
            
            call_args = mock_extract.call_args
            assert call_args[1]["generation_guidance"] == guidance

    @pytest.mark.asyncio
    async def test_extract_record_data_with_callback(self, api_instance, sample_schema, ai_config) -> None:
        """Test record data extraction with dataframe update callback."""
        api_instance.set_schema(sample_schema)
        api_instance.set_ai_configuration(ai_config)
        
        input_texts = ["Test text"]
        callback = MagicMock()
        
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_extractor.extract_record_data"
        ) as mock_extract:
            mock_extract.return_value = ({}, {})
            
            await api_instance.extract_record_data(input_texts, df_update_callback=callback)
            
            call_args = mock_extract.call_args
            assert call_args[1]["df_update_callback"] == callback

    @pytest.mark.asyncio
    async def test_extract_record_data_with_batch_callback(self, api_instance, sample_schema, ai_config) -> None:
        """Test record data extraction with batch callback."""
        api_instance.set_schema(sample_schema)
        api_instance.set_ai_configuration(ai_config)
        
        input_texts = ["Test text"]
        batch_callback = MagicMock()
        
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_extractor.extract_record_data"
        ) as mock_extract:
            mock_extract.return_value = ({}, {})
            
            await api_instance.extract_record_data(input_texts, callback_batch=batch_callback)
            
            call_args = mock_extract.call_args
            assert call_args[1]["callback_batch"] == batch_callback

    @pytest.mark.asyncio
    async def test_extract_record_data_empty_input(self, api_instance, sample_schema, ai_config) -> None:
        """Test record data extraction with empty input."""
        api_instance.set_schema(sample_schema)
        api_instance.set_ai_configuration(ai_config)
        
        with patch(
            "intelligence_toolkit.extract_record_data.api.data_extractor.extract_record_data"
        ) as mock_extract:
            mock_extract.return_value = ({}, {})
            
            await api_instance.extract_record_data([])
            
            mock_extract.assert_called_once()
