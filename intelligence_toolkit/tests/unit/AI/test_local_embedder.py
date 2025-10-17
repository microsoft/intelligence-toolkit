# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from intelligence_toolkit.AI.local_embedder import LocalEmbedder


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_local_embedder_initialization(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
            model="all-distilroberta-v1"
        )
        
        assert embedder.local_client is not None
        mock_st.assert_called_once_with("all-distilroberta-v1")


def test_local_embedder_initialization_default_model(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        from intelligence_toolkit.AI.defaults import DEFAULT_LOCAL_EMBEDDING_MODEL
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
        )
        
        mock_st.assert_called_once_with(DEFAULT_LOCAL_EMBEDDING_MODEL)


def test_local_embedder_initialization_none_model(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        from intelligence_toolkit.AI.defaults import DEFAULT_LOCAL_EMBEDDING_MODEL
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
            model=None
        )
        
        mock_st.assert_called_once_with(DEFAULT_LOCAL_EMBEDDING_MODEL)


def test_local_embedder_initialization_model_error(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        mock_st.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Failed to load local embedding model"):
            LocalEmbedder(
                db_name="test_embeddings",
                db_path=temp_db_path,
                model="invalid-model"
            )


def test_local_embedder_generate_embedding(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        # Mock the encode method
        import numpy as np
        mock_array = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_array
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
        )
        
        result = embedder._generate_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("test text")


@pytest.mark.asyncio
async def test_local_embedder_generate_embedding_async(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_array = np.array([0.4, 0.5, 0.6])
        mock_model.encode.return_value = mock_array
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
        )
        
        result = await embedder._generate_embedding_async("test text")
        
        assert result == [0.4, 0.5, 0.6]
        mock_model.encode.assert_called_once_with("test text")


def test_local_embedder_generate_embedding_list(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_array
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
        )
        
        result = embedder._generate_embedding(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


def test_local_embedder_check_token_count_disabled(temp_db_path):
    with patch("intelligence_toolkit.AI.local_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        embedder = LocalEmbedder(
            db_name="test_embeddings",
            db_path=temp_db_path,
        )
        
        # Verify check_token_count is disabled for local embedder
        assert embedder.check_token_count == False
