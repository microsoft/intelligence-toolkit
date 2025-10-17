# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from intelligence_toolkit.generate_mock_data.api import GenerateMockData


def test_generate_mock_data_initialization():
    gmd = GenerateMockData()
    
    assert gmd.json_schema == {}
    assert gmd.record_arrays == []
    assert gmd.json_object == {}
    assert gmd.array_dfs == {}


def test_set_schema():
    gmd = GenerateMockData()
    schema = {
        "properties": {
            "records": {"type": "array", "items": {"type": "object", "properties": {}}}
        }
    }
    
    gmd.set_schema(schema)
    
    assert gmd.json_schema == schema
    assert isinstance(gmd.record_arrays, list)


@patch("intelligence_toolkit.generate_mock_data.api.data_generator.extract_array_fields")
def test_set_schema_extracts_arrays(mock_extract):
    mock_extract.return_value = [["records"], ["items"]]
    
    gmd = GenerateMockData()
    schema = {"properties": {}}
    
    gmd.set_schema(schema)
    
    assert mock_extract.called
    assert gmd.record_arrays == [["records"], ["items"]]


def test_set_ai_configuration():
    gmd = GenerateMockData()
    ai_config = MagicMock()
    
    gmd.set_ai_configuration(ai_config)
    
    assert gmd.ai_configuration == ai_config


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.data_generator.generate_data")
async def test_generate_data_records_basic(mock_generate):
    mock_generate.return_value = (
        {"records": []},
        {"records": pd.DataFrame()},
    )
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    gmd.set_schema({"properties": {}})
    
    await gmd.generate_data_records(
        num_records_overall=10,
        records_per_batch=5,
        duplicate_records_per_batch=1,
        related_records_per_batch=1,
    )
    
    assert mock_generate.called
    assert isinstance(gmd.json_object, dict)
    assert isinstance(gmd.array_dfs, dict)


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.data_generator.generate_data")
async def test_generate_data_records_with_guidance(mock_generate):
    mock_generate.return_value = ({}, {})
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    gmd.set_schema({"properties": {}})
    
    await gmd.generate_data_records(
        num_records_overall=5,
        records_per_batch=5,
        duplicate_records_per_batch=0,
        related_records_per_batch=0,
        generation_guidance="Generate realistic data",
        temperature=0.7,
    )
    
    call_args = mock_generate.call_args
    assert call_args[1]["generation_guidance"] == "Generate realistic data"
    assert call_args[1]["temperature"] == 0.7


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.data_generator.generate_data")
async def test_generate_data_records_with_callbacks(mock_generate):
    mock_generate.return_value = ({}, {})
    df_callback = MagicMock()
    batch_callback = MagicMock()
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    gmd.set_schema({"properties": {}})
    
    await gmd.generate_data_records(
        num_records_overall=10,
        records_per_batch=5,
        duplicate_records_per_batch=1,
        related_records_per_batch=1,
        df_update_callback=df_callback,
        callback_batch=batch_callback,
    )
    
    call_args = mock_generate.call_args[1]
    assert call_args["df_update_callback"] == df_callback
    assert call_args["callback_batch"] == batch_callback


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.data_generator.generate_data")
async def test_generate_data_records_parallel_batches(mock_generate):
    mock_generate.return_value = ({}, {})
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    gmd.set_schema({"properties": {}})
    
    await gmd.generate_data_records(
        num_records_overall=50,
        records_per_batch=10,
        duplicate_records_per_batch=2,
        related_records_per_batch=2,
        parallel_batches=10,
    )
    
    call_args = mock_generate.call_args[1]
    assert call_args["parallel_batches"] == 10


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.text_generator.generate_text_data")
async def test_generate_text_data_basic(mock_generate):
    mock_generate.return_value = (
        ["Text 1", "Text 2"],
        pd.DataFrame({"mock_text": ["Text 1", "Text 2"]}),
    )
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    
    await gmd.generate_text_data(df)
    
    assert mock_generate.called
    assert hasattr(gmd, "text_list")
    assert hasattr(gmd, "text_df")
    assert isinstance(gmd.text_list, list)
    assert isinstance(gmd.text_df, pd.DataFrame)


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.text_generator.generate_text_data")
async def test_generate_text_data_converts_rows_to_json(mock_generate):
    mock_generate.return_value = (["Text"], pd.DataFrame())
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    
    df = pd.DataFrame({"col1": ["value1"], "col2": ["value2"]})
    
    await gmd.generate_text_data(df)
    
    # Should convert each row to JSON string
    call_args = mock_generate.call_args[1]
    input_texts = call_args["input_texts"]
    assert isinstance(input_texts, list)
    assert len(input_texts) == 1


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.text_generator.generate_text_data")
async def test_generate_text_data_with_parameters(mock_generate):
    mock_generate.return_value = ([], pd.DataFrame())
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    
    df = pd.DataFrame({"data": [1]})
    
    await gmd.generate_text_data(
        df,
        generation_guidance="Be concise",
        temperature=0.3,
    )
    
    call_args = mock_generate.call_args[1]
    assert call_args["generation_guidance"] == "Be concise"
    assert call_args["temperature"] == 0.3


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.text_generator.generate_text_data")
async def test_generate_text_data_with_callback(mock_generate):
    mock_generate.return_value = ([], pd.DataFrame())
    callback = MagicMock()
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    
    df = pd.DataFrame({"data": [1]})
    
    await gmd.generate_text_data(df, df_update_callback=callback)
    
    call_args = mock_generate.call_args[1]
    assert call_args["df_update_callback"] == callback


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.api.text_generator.generate_text_data")
async def test_generate_text_data_empty_dataframe(mock_generate):
    mock_generate.return_value = ([], pd.DataFrame())
    
    gmd = GenerateMockData()
    gmd.set_ai_configuration(MagicMock())
    
    df = pd.DataFrame()
    
    await gmd.generate_text_data(df)
    
    call_args = mock_generate.call_args[1]
    assert call_args["input_texts"] == []


def test_schema_persistence():
    gmd = GenerateMockData()
    schema1 = {"properties": {"field1": {}}}
    schema2 = {"properties": {"field2": {}}}
    
    gmd.set_schema(schema1)
    assert gmd.json_schema == schema1
    
    gmd.set_schema(schema2)
    assert gmd.json_schema == schema2
