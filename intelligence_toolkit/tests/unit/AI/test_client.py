# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intelligence_toolkit.AI.classes import LLMCallback
from intelligence_toolkit.AI.client import OpenAIClient
from intelligence_toolkit.AI.defaults import DEFAULT_EMBEDDING_MODEL
from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration


@pytest.fixture
def openai_config():
    return OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "OpenAI",
    })


@pytest.fixture
def azure_openai_config():
    return OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "Azure OpenAI",
        "api_base": "https://test.openai.azure.com",
        "api_version": "2024-02-01",
        "az_auth_type": "Azure Key",
    })


def test_openai_client_initialization_openai(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI") as mock_async_openai:
        
        client = OpenAIClient(openai_config)
        
        mock_openai.assert_called_once_with(api_key="test_key")
        mock_async_openai.assert_called_once_with(api_key="test_key")
        assert client.configuration == openai_config


def test_openai_client_initialization_azure_with_key(azure_openai_config):
    with patch("intelligence_toolkit.AI.client.AzureOpenAI") as mock_azure, \
         patch("intelligence_toolkit.AI.client.AsyncAzureOpenAI") as mock_async_azure:
        
        client = OpenAIClient(azure_openai_config)
        
        mock_azure.assert_called_once_with(
            api_version="2024-02-01",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test_key",
        )
        mock_async_azure.assert_called_once_with(
            api_version="2024-02-01",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test_key",
        )


def test_openai_client_initialization_azure_without_api_base():
    config = OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "Azure OpenAI",
        "api_base": None,
    })
    
    with pytest.raises(ValueError, match="api_base is required for Azure OpenAI client"):
        OpenAIClient(config)


def test_openai_client_initialization_azure_managed_identity():
    config = OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "Azure OpenAI",
        "api_base": "https://test.openai.azure.com",
        "api_version": "2024-02-01",
        "az_auth_type": "Managed Identity",
    })
    
    with patch("intelligence_toolkit.AI.client.DefaultAzureCredential") as mock_cred, \
         patch("intelligence_toolkit.AI.client.get_bearer_token_provider") as mock_token, \
         patch("intelligence_toolkit.AI.client.AzureOpenAI") as mock_azure, \
         patch("intelligence_toolkit.AI.client.AsyncAzureOpenAI") as mock_async_azure:
        
        mock_token.return_value = "mock_token_provider"
        
        client = OpenAIClient(config)
        
        mock_cred.assert_called_once()
        mock_token.assert_called_once_with(
            mock_cred.return_value,
            "https://cognitiveservices.azure.com/.default",
        )
        
        mock_azure.assert_called_once_with(
            api_version="2024-02-01",
            azure_ad_token_provider="mock_token_provider",
            azure_endpoint="https://test.openai.azure.com",
        )


def test_openai_client_generate_chat_non_streaming(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai_class, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(openai_config)
        messages = [{"role": "user", "content": "Hello"}]
        result = client.generate_chat(messages, stream=False)
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()


def test_openai_client_generate_chat_streaming_with_callbacks(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai_class, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock streaming response
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " World"
        
        mock_client.chat.completions.create.return_value = [chunk1, chunk2]
        
        callback = LLMCallback()
        client = OpenAIClient(openai_config)
        messages = [{"role": "user", "content": "Hello"}]
        result = client.generate_chat(messages, stream=True, callbacks=[callback])
        
        assert result == "Hello World"
        assert len(callback.response) > 0


def test_openai_client_generate_chat_exception_handling(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai_class, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        client = OpenAIClient(openai_config)
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="Problem in OpenAI response"):
            client.generate_chat(messages, stream=False)


def test_openai_client_generate_embedding(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai_class, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        client = OpenAIClient(openai_config)
        result = client.generate_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input="test text",
            model=DEFAULT_EMBEDDING_MODEL
        )


def test_openai_client_generate_chat_custom_params(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai_class, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(openai_config)
        messages = [{"role": "user", "content": "Hello"}]
        result = client.generate_chat(
            messages,
            stream=False,
            max_tokens=1000,
            temperature=0.5
        )
        
        assert result == "Test response"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_openai_client_generate_chat_async(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI") as mock_async_class:
        
        mock_async_client = MagicMock()
        mock_async_class.return_value = mock_async_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Async response"
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        client = OpenAIClient(openai_config)
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.generate_chat_async(messages, stream=False)
        
        assert result == "Async response"


@pytest.mark.asyncio
async def test_openai_client_generate_embedding_async(openai_config):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI") as mock_async_class:
        
        mock_async_client = MagicMock()
        mock_async_class.return_value = mock_async_client
        
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.4, 0.5, 0.6]
        mock_async_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        
        client = OpenAIClient(openai_config)
        result = await client.generate_embedding_async("test text")
        
        assert result == [0.4, 0.5, 0.6]
