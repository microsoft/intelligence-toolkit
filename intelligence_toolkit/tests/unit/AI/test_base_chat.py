# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intelligence_toolkit.AI.base_chat import BaseChat
from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration


@pytest.fixture
def base_chat_config():
    return OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "OpenAI",
    })


@pytest.fixture
def base_chat(base_chat_config):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        return BaseChat(base_chat_config)


def test_base_chat_initialization(base_chat_config):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        chat = BaseChat(base_chat_config, concurrent_coroutines=10)
        
        assert chat.configuration is not None
        assert chat.semaphore._value == 10


def test_base_chat_initialization_default_coroutines(base_chat_config):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        from intelligence_toolkit.AI.defaults import DEFAULT_CONCURRENT_COROUTINES
        
        chat = BaseChat(base_chat_config)
        assert chat.semaphore._value == DEFAULT_CONCURRENT_COROUTINES


@pytest.mark.asyncio
async def test_generate_text_async_success(base_chat):
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch.object(base_chat, 'generate_chat_async', new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Test response"
        
        result = await base_chat.generate_text_async(messages, None, False)
        
        assert result == "Test response"
        mock_chat.assert_called_once_with(messages=messages, stream=False)


@pytest.mark.asyncio
async def test_generate_text_async_with_callbacks(base_chat):
    messages = [{"role": "user", "content": "Hello"}]
    callback = MagicMock()
    
    with patch.object(base_chat, 'generate_chat_async', new_callable=AsyncMock) as mock_chat, \
         patch.object(base_chat, 'progress_callback') as mock_progress:
        
        mock_chat.return_value = "Test response"
        
        result = await base_chat.generate_text_async(messages, [callback], False)
        
        assert result == "Test response"
        mock_progress.assert_called_once()


@pytest.mark.asyncio
async def test_generate_text_async_exception(base_chat):
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch.object(base_chat, 'generate_chat_async', new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="Problem in OpenAI response"):
            await base_chat.generate_text_async(messages, None, False)


@pytest.mark.asyncio
async def test_generate_texts_async_multiple_messages(base_chat):
    messages_list = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "World"}],
        [{"role": "user", "content": "Test"}],
    ]
    
    with patch.object(base_chat, 'generate_text_async', new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = ["Response 1", "Response 2", "Response 3"]
        
        results = await base_chat.generate_texts_async(messages_list)
        
        assert len(results) == 3
        assert results[0] == "Response 1"
        assert results[1] == "Response 2"
        assert results[2] == "Response 3"
        assert base_chat.total_tasks == 3


@pytest.mark.asyncio
async def test_generate_texts_async_with_callbacks(base_chat):
    messages_list = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "World"}],
    ]
    
    callback = MagicMock()
    
    with patch.object(base_chat, 'generate_text_async', new_callable=AsyncMock) as mock_generate, \
         patch.object(base_chat, 'track_progress', new_callable=AsyncMock) as mock_track:
        
        mock_generate.side_effect = ["Response 1", "Response 2"]
        
        results = await base_chat.generate_texts_async(messages_list, callbacks=[callback])
        
        assert len(results) == 2
        mock_track.assert_called_once()


@pytest.mark.asyncio
async def test_generate_texts_async_with_kwargs(base_chat):
    messages_list = [
        [{"role": "user", "content": "Hello"}],
    ]
    
    with patch.object(base_chat, 'generate_text_async', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = "Response"
        
        await base_chat.generate_texts_async(
            messages_list,
            temperature=0.5,
            max_tokens=100
        )
        
        # Verify kwargs were passed
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 100
