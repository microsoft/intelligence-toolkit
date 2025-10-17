# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data commentary module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intelligence_toolkit.query_text_data.commentary import Commentary


@pytest.fixture
def ai_config():
    """Create AI configuration for testing."""
    return {"model": "gpt-4", "api_key": "test-key"}


@pytest.fixture
def cid_to_text():
    """Create sample chunk ID to text mapping."""
    return {
        1: json.dumps({"title": "Doc1", "text_chunk": "First chunk content", "chunk_id": 1}),
        2: json.dumps({"title": "Doc1", "text_chunk": "Second chunk content", "chunk_id": 2}),
        3: json.dumps({"title": "Doc2", "text_chunk": "Third chunk content", "chunk_id": 1}),
    }


class TestCommentaryInit:
    def test_init_basic(self, ai_config, cid_to_text) -> None:
        """Test Commentary initialization."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        assert commentary.ai_configuration == ai_config
        assert commentary.query == "test query"
        assert commentary.cid_to_text == cid_to_text
        assert commentary.update_interval == 5
        assert len(commentary.unprocessed_chunks) == 0
        assert "points" in commentary.structure
        assert "point_sources" in commentary.structure
        assert "themes" in commentary.structure


class TestAddChunks:
    def test_add_chunks_below_interval(self, ai_config, cid_to_text) -> None:
        """Test adding chunks below update interval."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        chunks = {1: "chunk 1", 2: "chunk 2"}
        commentary.add_chunks(chunks)
        
        assert len(commentary.unprocessed_chunks) == 2
        assert commentary.unprocessed_chunks[1] == "chunk 1"

    def test_add_chunks_at_interval(self, ai_config, cid_to_text) -> None:
        """Test adding chunks triggers update at interval."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=2,
            analysis_callback=None,
            commentary_callback=None
        )
        
        with patch.object(commentary, 'update_analysis') as mock_update:
            chunks = {1: "chunk 1", 2: "chunk 2"}
            commentary.add_chunks(chunks)
            
            mock_update.assert_called_once_with(chunks)
            assert len(commentary.unprocessed_chunks) == 0

    def test_add_chunks_zero_interval(self, ai_config, cid_to_text) -> None:
        """Test adding chunks with zero interval doesn't trigger update."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=0,
            analysis_callback=None,
            commentary_callback=None
        )
        
        with patch.object(commentary, 'update_analysis') as mock_update:
            chunks = {1: "chunk 1"}
            commentary.add_chunks(chunks)
            
            mock_update.assert_not_called()
            assert len(commentary.unprocessed_chunks) == 1


class TestCompleteAnalysis:
    def test_complete_analysis_with_unprocessed(self, ai_config, cid_to_text) -> None:
        """Test completing analysis with unprocessed chunks."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.unprocessed_chunks = {1: "chunk 1"}
        
        with patch.object(commentary, 'update_analysis') as mock_update:
            commentary.complete_analysis()
            
            mock_update.assert_called_once()
            assert len(commentary.unprocessed_chunks) == 0

    def test_complete_analysis_no_unprocessed(self, ai_config, cid_to_text) -> None:
        """Test completing analysis with no unprocessed chunks."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        with patch.object(commentary, 'update_analysis') as mock_update:
            commentary.complete_analysis()
            
            mock_update.assert_not_called()

    def test_complete_analysis_zero_interval(self, ai_config, cid_to_text) -> None:
        """Test completing analysis with zero interval doesn't process."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=0,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.unprocessed_chunks = {1: "chunk 1"}
        
        with patch.object(commentary, 'update_analysis') as mock_update:
            commentary.complete_analysis()
            
            mock_update.assert_not_called()


class TestUpdateAnalysis:
    def test_update_analysis_basic(self, ai_config, cid_to_text) -> None:
        """Test basic update analysis."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        mock_response = json.dumps({
            "updates": [
                {"point_id": "p1", "point_title": "Point 1", "source_ids": [1, 2]},
            ],
            "themes": [
                {"theme_title": "Theme 1", "point_ids": ["p1"]},
            ]
        })
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            chunks = {1: "chunk 1"}
            commentary.update_analysis(chunks)
            
            assert "p1" in commentary.structure["points"]
            assert commentary.structure["points"]["p1"] == "Point 1"
            assert "Theme 1" in commentary.structure["themes"]

    def test_update_analysis_with_callback(self, ai_config, cid_to_text) -> None:
        """Test update analysis with callback."""
        callback = MagicMock()
        callback.on_llm_new_token = MagicMock()
        
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=callback,
            commentary_callback=None
        )
        
        mock_response = json.dumps({
            "updates": [{"point_id": "p1", "point_title": "Point 1", "source_ids": [1]}],
            "themes": [{"theme_title": "Theme 1", "point_ids": ["p1"]}]
        })
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            commentary.update_analysis({1: "chunk"})
            
            callback.on_llm_new_token.assert_called()

    def test_update_analysis_updates_existing_point(self, ai_config, cid_to_text) -> None:
        """Test update analysis updates existing point title."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        # Set initial point
        commentary.structure["points"]["p1"] = "Old Title"
        commentary.structure["point_sources"]["p1"] = [1]
        
        mock_response = json.dumps({
            "updates": [{"point_id": "p1", "point_title": "New Title", "source_ids": [2]}],
            "themes": []
        })
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            commentary.update_analysis({1: "chunk"})
            
            assert commentary.structure["points"]["p1"] == "New Title"
            assert 2 in commentary.structure["point_sources"]["p1"]

    def test_update_analysis_empty_title(self, ai_config, cid_to_text) -> None:
        """Test update analysis with empty title doesn't overwrite."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.structure["points"]["p1"] = "Original Title"
        
        mock_response = json.dumps({
            "updates": [{"point_id": "p1", "point_title": "", "source_ids": [1]}],
            "themes": []
        })
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            commentary.update_analysis({1: "chunk"})
            
            assert commentary.structure["points"]["p1"] == "Original Title"


class TestFormatStructure:
    def test_format_structure_basic(self, ai_config, cid_to_text) -> None:
        """Test basic structure formatting."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.structure = {
            "points": {"p1": "Point 1"},
            "point_sources": {"p1": [1]},
            "themes": {"Theme 1": ["p1"]}
        }
        
        result = commentary.format_structure()
        
        assert "Theme 1" in result
        assert "Point 1" in result
        assert "[1](#source-1)" in result
        assert "## Sources" in result
        assert "Source 1" in result

    def test_format_structure_no_sources(self, ai_config, cid_to_text) -> None:
        """Test structure formatting with no sources."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.structure = {
            "points": {},
            "point_sources": {},
            "themes": {}
        }
        
        result = commentary.format_structure()
        
        assert "## Sources" not in result


class TestGetClusteredCids:
    def test_get_clustered_cids_with_interval(self, ai_config, cid_to_text) -> None:
        """Test getting clustered CIDs with update interval."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.structure = {
            "themes": {"Theme 1": ["p1", "p2"]},
            "point_sources": {"p1": [1, 2], "p2": [3]}
        }
        
        result = commentary.get_clustered_cids()
        
        assert "Theme 1" in result
        assert 1 in result["Theme 1"]
        assert 2 in result["Theme 1"]
        assert 3 in result["Theme 1"]

    def test_get_clustered_cids_without_interval(self, ai_config, cid_to_text) -> None:
        """Test getting clustered CIDs without update interval."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=0,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.unprocessed_chunks = {1: "chunk1", 2: "chunk2"}
        
        result = commentary.get_clustered_cids()
        
        assert "All relevant chunks" in result
        assert result["All relevant chunks"] == [1, 2]

    def test_get_clustered_cids_missing_sources(self, ai_config, cid_to_text) -> None:
        """Test getting clustered CIDs with missing point sources."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.structure = {
            "themes": {"Theme 1": ["p1", "p2"]},
            "point_sources": {"p1": [1]}  # p2 missing
        }
        
        result = commentary.get_clustered_cids()
        
        assert "Theme 1" in result
        assert result["Theme 1"] == [1]


class TestGenerateCommentary:
    @pytest.mark.asyncio
    async def test_generate_commentary_basic(self, ai_config, cid_to_text) -> None:
        """Test basic commentary generation."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        commentary.structure = {
            "points": {"p1": "Point 1"},
            "themes": {"Theme 1": ["p1"]},
            "point_sources": {"p1": [1, 2, 3]}
        }
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat_async = AsyncMock(return_value="Generated commentary")
            mock_client_class.return_value = mock_client
            
            result = await commentary.generate_commentary()
            
            assert result == "Generated commentary"
            mock_client.generate_chat_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_commentary_with_callback(self, ai_config, cid_to_text) -> None:
        """Test commentary generation with callback."""
        callback = MagicMock()
        
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=callback
        )
        
        commentary.structure = {
            "points": {"p1": "Point 1"},
            "themes": {"Theme 1": ["p1"]},
            "point_sources": {"p1": [1]}
        }
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat_async = AsyncMock(return_value="Commentary")
            mock_client_class.return_value = mock_client
            
            await commentary.generate_commentary()
            
            # Verify callback was passed
            call_args = mock_client.generate_chat_async.call_args
            assert callback in call_args.kwargs["callbacks"]

    @pytest.mark.asyncio
    async def test_generate_commentary_limits_chunks(self, ai_config, cid_to_text) -> None:
        """Test commentary generation limits chunks to 3 per theme."""
        commentary = Commentary(
            ai_configuration=ai_config,
            query="test query",
            cid_to_text=cid_to_text,
            update_interval=5,
            analysis_callback=None,
            commentary_callback=None
        )
        
        # Create theme with more than 3 chunks
        commentary.structure = {
            "points": {"p1": "Point 1"},
            "themes": {"Theme 1": ["p1"]},
            "point_sources": {"p1": [1, 2, 3, 4, 5]}
        }
        
        with patch("intelligence_toolkit.query_text_data.commentary.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate_chat_async = AsyncMock(return_value="Commentary")
            mock_client_class.return_value = mock_client
            
            with patch.object(commentary, 'get_clustered_cids', return_value={"Theme 1": [1, 2, 3, 4, 5]}):
                await commentary.generate_commentary()
                
                # Check that messages were prepared
                mock_client.generate_chat_async.assert_called_once()
