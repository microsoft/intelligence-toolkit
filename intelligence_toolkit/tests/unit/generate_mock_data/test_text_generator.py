# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from intelligence_toolkit.generate_mock_data.text_generator import (
    generate_text_data,
)


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.text_generator.tqdm_asyncio.gather")
@patch("intelligence_toolkit.generate_mock_data.text_generator._generate_text_async")
async def test_generate_text_data_basic(mock_generate, mock_gather):
    mock_generate.return_value = "Generated text"
    mock_gather.return_value = ["Text 1", "Text 2", "Text 3"]
    
    ai_config = MagicMock()
    input_texts = ["input1", "input2", "input3"]
    
    texts, df = await generate_text_data(ai_config, input_texts)
    
    assert isinstance(texts, list)
    assert isinstance(df, pd.DataFrame)
    assert len(texts) == 3
    assert "mock_text" in df.columns


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.text_generator.tqdm_asyncio.gather")
@patch("intelligence_toolkit.generate_mock_data.text_generator._generate_text_async")
async def test_generate_text_data_with_callback(mock_generate, mock_gather):
    mock_gather.return_value = ["Generated text"]
    callback = MagicMock()
    
    ai_config = MagicMock()
    input_texts = ["input"]
    
    texts, df = await generate_text_data(
        ai_config, input_texts, df_update_callback=callback
    )
    
    assert callback.called


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.text_generator.tqdm_asyncio.gather")
async def test_generate_text_data_empty_input(mock_gather):
    mock_gather.return_value = []
    
    ai_config = MagicMock()
    input_texts = []
    
    texts, df = await generate_text_data(ai_config, input_texts)
    
    assert texts == []
    assert len(df) == 0


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.text_generator.tqdm_asyncio.gather")
@patch("intelligence_toolkit.generate_mock_data.text_generator._generate_text_async")
async def test_generate_text_data_with_parameters(mock_generate, mock_gather):
    mock_gather.return_value = ["Text"]
    
    ai_config = MagicMock()
    input_texts = ["input"]
    generation_guidance = "Be creative"
    temperature = 0.8
    
    texts, df = await generate_text_data(
        ai_config,
        input_texts,
        generation_guidance=generation_guidance,
        temperature=temperature,
    )
    
    assert isinstance(texts, list)
    assert len(texts) == 1


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.text_generator.tqdm_asyncio.gather")
@patch("intelligence_toolkit.generate_mock_data.text_generator._generate_text_async")
async def test_generate_text_data_batching(mock_generate, mock_gather):
    # Test with more inputs than default parallelism
    # Mock returns texts for first batch, then second batch
    mock_gather.side_effect = [["Text"] * 10, ["Text"] * 5]
    
    ai_config = MagicMock()
    input_texts = ["input"] * 15
    
    texts, df = await generate_text_data(ai_config, input_texts, parallelism=10)
    
    # Should batch into groups
    assert len(texts) == 15
    assert len(df) == 15


@pytest.mark.asyncio
@patch("intelligence_toolkit.generate_mock_data.text_generator.utils.generate_text_async")
@patch("intelligence_toolkit.generate_mock_data.text_generator.utils.prepare_messages")
async def test_generate_text_async_internal(mock_prepare, mock_generate):
    mock_prepare.return_value = [{"role": "user", "content": "test"}]
    mock_generate.return_value = "Generated response"
    
    from intelligence_toolkit.generate_mock_data.text_generator import _generate_text_async
    
    ai_config = MagicMock()
    result = await _generate_text_async(ai_config, "input", "guidance", 0.7)
    
    assert result == "Generated response"
    assert mock_prepare.called
    assert mock_generate.called
