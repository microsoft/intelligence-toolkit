# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data helper_functions module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intelligence_toolkit.query_text_data.helper_functions import (
    embed_queries,
    embed_texts,
    get_adjacent_chunks,
    get_test_progress,
    parse_history_elements,
)


class TestGetAdjacentChunks:
    def test_get_adjacent_chunks_both_directions(self) -> None:
        """Test getting adjacent chunks in both directions."""
        previous = {3: 2, 2: 1}
        next_ = {1: 2, 2: 3}
        
        result = get_adjacent_chunks(2, previous, next_, steps=1)
        
        assert 1 in result
        assert 3 in result
        assert len(result) == 2

    def test_get_adjacent_chunks_multiple_steps(self) -> None:
        """Test getting adjacent chunks with multiple steps."""
        previous = {4: 3, 3: 2, 2: 1}
        next_ = {1: 2, 2: 3, 3: 4}
        
        result = get_adjacent_chunks(2, previous, next_, steps=2)
        
        assert 1 in result
        assert 3 in result
        assert 4 in result
        assert len(result) == 3

    def test_get_adjacent_chunks_at_boundary(self) -> None:
        """Test getting adjacent chunks at sequence boundary."""
        previous = {2: 1}
        next_ = {1: 2}
        
        # At the start
        result = get_adjacent_chunks(1, previous, next_, steps=2)
        assert 2 in result
        assert len(result) == 1

    def test_get_adjacent_chunks_isolated(self) -> None:
        """Test getting adjacent chunks for isolated chunk."""
        previous = {}
        next_ = {}
        
        result = get_adjacent_chunks(5, previous, next_, steps=1)
        
        assert len(result) == 0

    def test_get_adjacent_chunks_zero_steps(self) -> None:
        """Test getting adjacent chunks with zero steps."""
        previous = {2: 1}
        next_ = {1: 2}
        
        result = get_adjacent_chunks(1, previous, next_, steps=0)
        
        assert len(result) == 0


class TestGetTestProgress:
    def test_get_test_progress_single_search(self) -> None:
        """Test test progress with single search."""
        history = [
            ("search1", 1, "Yes"),
            ("search1", 2, "No"),
            ("search1", 3, "Yes"),
        ]
        
        result = get_test_progress(history)
        
        assert "2/3" in result  # 2 relevant out of 3 tested
        assert "search1" in result

    def test_get_test_progress_multiple_searches(self) -> None:
        """Test test progress with multiple searches."""
        history = [
            ("search1", 1, "Yes"),
            ("search1", 2, "Yes"),
            ("search2", 3, "No"),
            ("search2", 4, "Yes"),
        ]
        
        result = get_test_progress(history)
        
        assert "3/4" in result  # Total: 3 relevant out of 4 tested
        assert "search1: 2/2" in result
        assert "search2: 1/2" in result

    def test_get_test_progress_no_relevant_chunks(self) -> None:
        """Test test progress when search finds no relevant chunks."""
        history = [
            ("search1", 1, "No"),
            ("search1", 2, "No"),
        ]
        
        result = get_test_progress(history)
        
        assert "0/2" in result
        assert "color: red" in result  # Should mark failed searches

    def test_get_test_progress_empty_history(self) -> None:
        """Test test progress with empty history."""
        result = get_test_progress([])
        
        assert "0/0" in result

    def test_get_test_progress_all_relevant(self) -> None:
        """Test test progress when all chunks are relevant."""
        history = [
            ("search1", 1, "Yes"),
            ("search1", 2, "Yes"),
            ("search1", 3, "Yes"),
        ]
        
        result = get_test_progress(history)
        
        assert "3/3" in result
        assert "search1: 3/3" in result
        assert "color: red" not in result


class TestParseHistoryElements:
    def test_parse_history_elements_basic(self) -> None:
        """Test extracting elements from test history."""
        history = [
            ("search1", 1, "Yes"),
            ("search1", 2, "No"),
            ("search1", 3, "Yes"),
        ]
        previous = {2: 1, 3: 2}
        next_ = {1: 2, 2: 3}
        
        relevant, seen, adjacent = parse_history_elements(history, previous, next_, 1)
        
        assert 1 in relevant
        assert 3 in relevant
        assert len(relevant) == 2
        
        assert 1 in seen
        assert 2 in seen
        assert 3 in seen
        assert len(seen) == 3

    def test_parse_history_elements_finds_adjacent(self) -> None:
        """Test that adjacent chunks are identified correctly."""
        history = [
            ("search1", 2, "Yes"),
        ]
        previous = {4: 3, 3: 2, 2: 1}
        next_ = {1: 2, 2: 3, 3: 4}
        
        relevant, seen, adjacent = parse_history_elements(history, previous, next_, 2)
        
        # Chunk 2 is relevant, so chunks 1, 3, 4 should be adjacent
        assert 1 in adjacent or 3 in adjacent or 4 in adjacent
        # Already seen chunks should not be in adjacent list
        assert 2 not in adjacent

    def test_parse_history_elements_excludes_seen_from_adjacent(self) -> None:
        """Test that already seen chunks are excluded from adjacent list."""
        history = [
            ("search1", 2, "Yes"),
            ("search1", 3, "No"),
        ]
        previous = {3: 2, 2: 1}
        next_ = {1: 2, 2: 3}
        
        relevant, seen, adjacent = parse_history_elements(history, previous, next_, 1)
        
        # Chunk 3 is adjacent to 2 but already seen, so not in adjacent list
        assert 3 not in adjacent
        assert 3 in seen

    def test_parse_history_elements_empty_history(self) -> None:
        """Test with empty history."""
        relevant, seen, adjacent = parse_history_elements([], {}, {}, 1)
        
        assert len(relevant) == 0
        assert len(seen) == 0
        assert len(adjacent) == 0


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_embed_texts_basic(self) -> None:
        """Test basic text embedding."""
        cid_to_text = {
            1: "text one",
            2: "text two",
        }
        
        mock_embedder = MagicMock()
        mock_embedder.embed_store_many = AsyncMock(return_value=[
            {
                "hash": "hash1",
                "vector": [0.1, 0.2, 0.3],
                "additional_details": '{"cid": 1}'
            },
            {
                "hash": "hash2",
                "vector": [0.4, 0.5, 0.6],
                "additional_details": '{"cid": 2}'
            },
        ])
        
        result = await embed_texts(cid_to_text, mock_embedder)
        
        assert 1 in result
        assert 2 in result
        assert result[1] == [0.1, 0.2, 0.3]
        assert result[2] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_texts_with_callbacks(self) -> None:
        """Test text embedding with callbacks."""
        cid_to_text = {1: "text"}
        mock_embedder = MagicMock()
        mock_embedder.embed_store_many = AsyncMock(return_value=[
            {
                "hash": "hash1",
                "vector": [0.1],
                "additional_details": '{"cid": 1}'
            },
        ])
        callbacks = [MagicMock()]
        
        await embed_texts(cid_to_text, mock_embedder, callbacks=callbacks)
        
        mock_embedder.embed_store_many.assert_called_once()
        call_args = mock_embedder.embed_store_many.call_args
        assert call_args[0][1] == callbacks

    @pytest.mark.asyncio
    async def test_embed_texts_skips_empty_details(self) -> None:
        """Test that items with empty additional_details are skipped."""
        cid_to_text = {1: "text"}
        mock_embedder = MagicMock()
        mock_embedder.embed_store_many = AsyncMock(return_value=[
            {
                "hash": "hash1",
                "vector": [0.1],
                "additional_details": '{}'  # Empty details
            },
        ])
        
        result = await embed_texts(cid_to_text, mock_embedder)
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_embed_texts_cache_control(self) -> None:
        """Test cache_data parameter is passed correctly."""
        cid_to_text = {1: "text"}
        mock_embedder = MagicMock()
        mock_embedder.embed_store_many = AsyncMock(return_value=[
            {
                "hash": "hash1",
                "vector": [0.1],
                "additional_details": '{"cid": 1}'
            },
        ])
        
        await embed_texts(cid_to_text, mock_embedder, cache_data=False)
        
        call_args = mock_embedder.embed_store_many.call_args
        assert call_args[0][2] is False


class TestEmbedQueries:
    @pytest.mark.asyncio
    async def test_embed_queries_basic(self) -> None:
        """Test basic query embedding."""
        qid_to_text = {
            "q1": "query one",
            "q2": "query two",
        }
        
        mock_embedder = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.helper_functions.hash_text") as mock_hash:
            # Make hash_text return predictable hashes
            mock_hash.side_effect = lambda x: f"hash_{x}"
            
            mock_embedder.embed_store_many = AsyncMock(return_value=[
                {
                    "hash": "hash_query one",
                    "vector": [0.1, 0.2],
                    "additional_details": '{"qid": "q1"}'
                },
                {
                    "hash": "hash_query two",
                    "vector": [0.3, 0.4],
                    "additional_details": '{"qid": "q2"}'
                },
            ])
            
            result = await embed_queries(qid_to_text, mock_embedder)
            
            assert "q1" in result
            assert "q2" in result
            assert result["q1"] == [0.1, 0.2]
            assert result["q2"] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_embed_queries_handles_mismatched_details(self) -> None:
        """Test query embedding handles cases where stored details don't match."""
        qid_to_text = {"q1": "query"}
        
        mock_embedder = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.helper_functions.hash_text") as mock_hash:
            mock_hash.return_value = "hash1"
            
            # Simulate case where additional_details in embedded_data doesn't have qid
            mock_embedder.embed_store_many = AsyncMock(return_value=[
                {
                    "hash": "hash1",
                    "vector": [0.1],
                    "additional_details": '{}'  # Missing qid but we'll use original data
                },
            ])
            
            result = await embed_queries(qid_to_text, mock_embedder)
            
            # Should return the query with vector using qid from original request
            assert "q1" in result

    @pytest.mark.asyncio
    async def test_embed_queries_with_callbacks(self) -> None:
        """Test query embedding with callbacks."""
        qid_to_text = {"q1": "query"}
        mock_embedder = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.helper_functions.hash_text") as mock_hash:
            mock_hash.return_value = "hash1"
            
            mock_embedder.embed_store_many = AsyncMock(return_value=[
                {
                    "hash": "hash1",
                    "vector": [0.1],
                    "additional_details": '{"qid": "q1"}'
                },
            ])
            callbacks = [MagicMock()]
            
            await embed_queries(qid_to_text, mock_embedder, callbacks=callbacks)
            
            call_args = mock_embedder.embed_store_many.call_args
            assert call_args[0][1] == callbacks

    @pytest.mark.asyncio
    async def test_embed_queries_no_matching_data(self) -> None:
        """Test query embedding when no matching data item found."""
        qid_to_text = {"q1": "query"}
        
        mock_embedder = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.helper_functions.hash_text") as mock_hash:
            mock_hash.return_value = "hash1"
            
            mock_embedder.embed_store_many = AsyncMock(return_value=[
                {
                    "hash": "different_hash",  # Hash that won't match
                    "vector": [0.1],
                    "additional_details": '{"qid": "q1"}'
                },
            ])
            
            # Should handle gracefully and not crash
            result = await embed_queries(qid_to_text, mock_embedder)
            
            # Won't have the result since hash doesn't match
            assert len(result) == 0
