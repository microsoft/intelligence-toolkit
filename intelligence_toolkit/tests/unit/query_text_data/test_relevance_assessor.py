# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data relevance_assessor module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from intelligence_toolkit.query_text_data.relevance_assessor import (
    assess_relevance,
    process_relevance_responses,
)


class TestProcessRelevanceResponses:
    def test_process_relevance_responses_basic(self) -> None:
        """Test basic relevance response processing."""
        test_history = []
        cid_to_text = {
            1: json.dumps({"title": "Doc1", "text_chunk": "Chunk 1", "chunk_id": 1}),
            2: json.dumps({"title": "Doc1", "text_chunk": "Chunk 2", "chunk_id": 2}),
        }
        
        num_relevant = process_relevance_responses(
            search_label="test",
            search_cids=[1, 2],
            cid_to_text=cid_to_text,
            mapped_responses=["Yes", "No"],
            test_history=test_history,
            progress_callback=None,
            chunk_callback=None,
            commentary=None
        )
        
        assert num_relevant == 1
        assert len(test_history) == 2
        assert test_history[0] == ("test", 1, "Yes")
        assert test_history[1] == ("test", 2, "No")

    def test_process_relevance_responses_with_callbacks(self) -> None:
        """Test relevance response processing with callbacks."""
        test_history = []
        cid_to_text = {1: json.dumps({"text_chunk": "Chunk"})}
        progress_callback = MagicMock()
        chunk_callback = MagicMock()
        
        process_relevance_responses(
            search_label="test",
            search_cids=[1],
            cid_to_text=cid_to_text,
            mapped_responses=["Yes"],
            test_history=test_history,
            progress_callback=progress_callback,
            chunk_callback=chunk_callback,
            commentary=None
        )
        
        progress_callback.assert_called_once()
        chunk_callback.assert_called_once()

    def test_process_relevance_responses_with_commentary(self) -> None:
        """Test relevance response processing with commentary."""
        test_history = []
        cid_to_text = {1: json.dumps({"text_chunk": "Chunk"})}
        commentary = MagicMock()
        commentary.add_chunks = MagicMock()
        
        process_relevance_responses(
            search_label="test",
            search_cids=[1],
            cid_to_text=cid_to_text,
            mapped_responses=["Yes"],
            test_history=test_history,
            progress_callback=None,
            chunk_callback=None,
            commentary=commentary
        )
        
        commentary.add_chunks.assert_called_once()

    def test_process_relevance_responses_skips_duplicates(self) -> None:
        """Test that duplicate CIDs are skipped."""
        test_history = [("test", 1, "Yes")]
        cid_to_text = {1: json.dumps({"text_chunk": "Chunk"})}
        
        num_relevant = process_relevance_responses(
            search_label="test",
            search_cids=[1],
            cid_to_text=cid_to_text,
            mapped_responses=["Yes"],
            test_history=test_history,
            progress_callback=None,
            chunk_callback=None,
            commentary=None
        )
        
        assert num_relevant == 0
        assert len(test_history) == 1

    def test_process_relevance_responses_no_relevant(self) -> None:
        """Test processing with no relevant responses."""
        test_history = []
        cid_to_text = {1: json.dumps({"text_chunk": "Chunk 1"}), 2: json.dumps({"text_chunk": "Chunk 2"})}
        
        num_relevant = process_relevance_responses(
            search_label="test",
            search_cids=[1, 2],
            cid_to_text=cid_to_text,
            mapped_responses=["No", "No"],
            test_history=test_history,
            progress_callback=None,
            chunk_callback=None,
            commentary=None
        )
        
        assert num_relevant == 0


class TestAssessRelevance:
    @pytest.mark.asyncio
    async def test_assess_relevance_basic(self) -> None:
        """Test basic relevance assessment."""
        cid_to_text = {
            1: json.dumps({"title": "Doc", "text_chunk": "Test chunk", "chunk_id": 1})
        }
        test_history = []
        
        with patch("intelligence_toolkit.query_text_data.relevance_assessor.utils") as mock_utils:
            mock_utils.prepare_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
            mock_utils.map_generate_text = AsyncMock(return_value=["Yes"])
            
            result = await assess_relevance(
                ai_configuration={"model": "gpt-4"},
                search_label="test",
                search_cids=[1],
                cid_to_text=cid_to_text,
                query="test query",
                logit_bias={},
                relevance_test_budget=100,
                num_adjacent=0,
                relevance_test_batch_size=10,
                test_history=test_history,
                progress_callback=None,
                chunk_callback=None,
                commentary=None
            )
            
            assert result is True
            assert len(test_history) == 1

    @pytest.mark.asyncio
    async def test_assess_relevance_no_relevant(self) -> None:
        """Test relevance assessment with no relevant chunks."""
        cid_to_text = {
            1: json.dumps({"title": "Doc", "text_chunk": "Test chunk", "chunk_id": 1})
        }
        test_history = []
        
        with patch("intelligence_toolkit.query_text_data.relevance_assessor.utils") as mock_utils:
            mock_utils.prepare_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
            mock_utils.map_generate_text = AsyncMock(return_value=["No"])
            
            result = await assess_relevance(
                ai_configuration={"model": "gpt-4"},
                search_label="test",
                search_cids=[1],
                cid_to_text=cid_to_text,
                query="test query",
                logit_bias={},
                relevance_test_budget=100,
                num_adjacent=0,
                relevance_test_batch_size=10,
                test_history=test_history,
                progress_callback=None,
                chunk_callback=None,
                commentary=None
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_assess_relevance_truncates_batch_at_budget(self) -> None:
        """Test that relevance assessment truncates batch when approaching budget."""
        cid_to_text = {i: json.dumps({"title": f"Doc{i}", "text_chunk": f"Chunk {i}", "chunk_id": i}) for i in range(1, 12)}
        test_history = [("prev", i, "Yes") for i in range(1, 9)]  # 8 existing tests
        
        with patch("intelligence_toolkit.query_text_data.relevance_assessor.utils") as mock_utils:
            mock_utils.prepare_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
            # Mock should return fewer results if batch is truncated
            mock_utils.map_generate_text = AsyncMock(return_value=["Yes", "Yes"])
            
            await assess_relevance(
                ai_configuration={"model": "gpt-4"},
                search_label="test",
                search_cids=list(range(9, 12)),  # 3 new CIDs
                cid_to_text=cid_to_text,
                query="test query",
                logit_bias={},
                relevance_test_budget=10,  # Budget of 10, with 8 existing = 2 remaining
                num_adjacent=0,
                relevance_test_batch_size=10,
                test_history=test_history,
                progress_callback=None,
                chunk_callback=None,
                commentary=None
            )
            
            # Should have truncated the batch to fit budget
            assert len(test_history) == 10

    @pytest.mark.asyncio
    async def test_assess_relevance_batch_processing(self) -> None:
        """Test relevance assessment processes in batches."""
        cid_to_text = {i: json.dumps({"title": f"Doc{i}", "text_chunk": f"Chunk {i}", "chunk_id": i}) for i in range(1, 8)}
        test_history = []
        
        with patch("intelligence_toolkit.query_text_data.relevance_assessor.utils") as mock_utils:
            mock_utils.prepare_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
            # First batch: Yes, second batch: No (should stop early)
            mock_utils.map_generate_text = AsyncMock(side_effect=[
                ["Yes", "Yes", "Yes"],
                ["No", "No", "No"]
            ])
            
            result = await assess_relevance(
                ai_configuration={"model": "gpt-4"},
                search_label="test",
                search_cids=list(range(1, 8)),
                cid_to_text=cid_to_text,
                query="test query",
                logit_bias={},
                relevance_test_budget=100,
                num_adjacent=0,
                relevance_test_batch_size=3,
                test_history=test_history,
                progress_callback=None,
                chunk_callback=None,
                commentary=None
            )
            
            # Should stop after second batch returns all "No"
            assert result is False
            assert 3 < len(test_history) <= 6

    @pytest.mark.asyncio
    async def test_assess_relevance_with_callbacks(self) -> None:
        """Test relevance assessment triggers callbacks."""
        cid_to_text = {1: json.dumps({"title": "Doc", "text_chunk": "Chunk", "chunk_id": 1})}
        test_history = []
        progress_callback = MagicMock()
        chunk_callback = MagicMock()
        commentary = MagicMock()
        commentary.add_chunks = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.relevance_assessor.utils") as mock_utils:
            mock_utils.prepare_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
            mock_utils.map_generate_text = AsyncMock(return_value=["Yes"])
            
            await assess_relevance(
                ai_configuration={"model": "gpt-4"},
                search_label="test",
                search_cids=[1],
                cid_to_text=cid_to_text,
                query="test query",
                logit_bias={},
                relevance_test_budget=100,
                num_adjacent=0,
                relevance_test_batch_size=10,
                test_history=test_history,
                progress_callback=progress_callback,
                chunk_callback=chunk_callback,
                commentary=commentary
            )
            
            progress_callback.assert_called()
            chunk_callback.assert_called()
            commentary.add_chunks.assert_called()
