# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration
from intelligence_toolkit.AI.openai_embedder import OpenAIEmbedder


@pytest.fixture
def openai_config():
    return OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "OpenAI",
        "embedding_model": "text-embedding-3-small",
    })


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_openai_embedder_initialization(openai_config, temp_db_path):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        embedder = OpenAIEmbedder(
            openai_config,
            db_name="test_embeddings",
            db_path=temp_db_path
        )
        
        assert embedder.configuration == openai_config
        assert embedder.openai_client is not None
        assert embedder.vector_store is not None


def test_openai_embedder_generate_embedding(openai_config, temp_db_path):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        embedder = OpenAIEmbedder(
            openai_config,
            db_name="test_embeddings",
            db_path=temp_db_path
        )
        
        result = embedder._generate_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_embedder_generate_embedding_async(openai_config, temp_db_path):
    with patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI") as mock_async_openai:
        
        mock_async_client = MagicMock()
        mock_async_openai.return_value = mock_async_client
        
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.4, 0.5, 0.6]
        mock_async_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        
        embedder = OpenAIEmbedder(
            openai_config,
            db_name="test_embeddings",
            db_path=temp_db_path
        )
        
        result = await embedder._generate_embedding_async("test text")
        
        assert result == [0.4, 0.5, 0.6]
        mock_async_client.embeddings.create.assert_called_once()


def test_openai_embedder_uses_configured_model(openai_config, temp_db_path):
    with patch("intelligence_toolkit.AI.client.OpenAI") as mock_openai, \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        embedder = OpenAIEmbedder(
            openai_config,
            db_name="test_embeddings",
            db_path=temp_db_path
        )
        
        embedder._generate_embedding("test text")
        
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-small"
