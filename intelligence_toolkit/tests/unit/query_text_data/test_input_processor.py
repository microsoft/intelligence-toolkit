# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data input_processor module."""

import json
from unittest.mock import MagicMock, patch

from intelligence_toolkit.query_text_data.input_processor import (
    PeriodOption,
    concert_titled_texts_to_chunks,
    process_chunks,
    process_json_text,
    process_json_texts,
)


class TestConcertTitledTextsToChunks:
    def test_concert_titled_texts_basic(self) -> None:
        """Test converting titled texts to chunks."""
        titled_texts = {
            "doc1": "This is a test document with some content.",
            "doc2": "Another document with different content."
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.side_effect = lambda x: [x]  # Don't actually split
            mock_splitter.return_value = mock_instance
            
            result = concert_titled_texts_to_chunks(titled_texts)
            
            assert len(result) > 0
            # Should have processed the documents
            assert mock_instance.split.call_count >= 1


class TestProcessJsonText:
    def test_process_json_text_no_period(self) -> None:
        """Test processing JSON text without period."""
        text_json = {
            "title": "Test Document",
            "text": "This is test content that will be chunked."
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk 1", "Chunk 2"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.NONE)
            
            assert len(result) == 2
            # Verify chunks are JSON strings
            chunk1 = json.loads(result[0])
            assert chunk1["title"] == "Test Document"
            assert chunk1["chunk_id"] == 1
            assert chunk1["text_chunk"] == "Chunk 1"

    def test_process_json_text_with_day_period(self) -> None:
        """Test processing JSON text with DAY period."""
        text_json = {
            "title": "Test Document",
            "text": "Test content",
            "timestamp": "2024-01-15T10:30:00"
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk 1"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.DAY)
            
            assert len(result) == 1
            chunk = json.loads(result[0])
            assert "period" in chunk
            assert chunk["period"] == "2024-01-15"
            assert chunk["timestamp"] == "2024-01-15T10:30:00"

    def test_process_json_text_with_week_period(self) -> None:
        """Test processing JSON text with WEEK period."""
        text_json = {
            "title": "Test",
            "text": "Content",
            "timestamp": "2024-01-15T10:00:00"
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.WEEK)
            
            chunk = json.loads(result[0])
            assert "period" in chunk
            assert "2024" in chunk["period"]

    def test_process_json_text_with_month_period(self) -> None:
        """Test processing JSON text with MONTH period."""
        text_json = {
            "title": "Test",
            "text": "Content",
            "timestamp": "2024-01-15T10:00:00"
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.MONTH)
            
            chunk = json.loads(result[0])
            assert chunk["period"] == "2024-01"

    def test_process_json_text_with_quarter_period(self) -> None:
        """Test processing JSON text with QUARTER period."""
        text_json = {
            "title": "Test",
            "text": "Content",
            "timestamp": "2024-02-15T10:00:00"  # February = Q1
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.QUARTER)
            
            chunk = json.loads(result[0])
            assert chunk["period"] == "2024-Q1"

    def test_process_json_text_with_year_period(self) -> None:
        """Test processing JSON text with YEAR period."""
        text_json = {
            "title": "Test",
            "text": "Content",
            "timestamp": "2024-06-15T10:00:00"
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.YEAR)
            
            chunk = json.loads(result[0])
            assert chunk["period"] == "2024"

    def test_process_json_text_with_metadata(self) -> None:
        """Test processing JSON text with metadata."""
        text_json = {
            "title": "Test",
            "text": "Content",
            "metadata": {"author": "Test Author", "tags": ["test", "sample"]}
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.NONE)
            
            chunk = json.loads(result[0])
            assert "metadata" in chunk
            assert chunk["metadata"]["author"] == "Test Author"
            assert "test" in chunk["metadata"]["tags"]

    def test_process_json_text_multiple_chunks(self) -> None:
        """Test processing JSON text that splits into multiple chunks."""
        text_json = {
            "title": "Long Document",
            "text": "This is a long document."
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_text(text_json, PeriodOption.NONE)
            
            assert len(result) == 3
            # Verify chunk IDs are sequential
            for i, chunk_str in enumerate(result):
                chunk = json.loads(chunk_str)
                assert chunk["chunk_id"] == i + 1


class TestProcessJsonTexts:
    def test_process_json_texts_single_file(self) -> None:
        """Test processing JSON texts from single file."""
        file_to_texts = {
            "file1.txt": {
                "title": "Doc1",
                "text": "Content 1"
            }
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_texts(file_to_texts, PeriodOption.NONE)
            
            assert "file1.txt" in result
            assert len(result["file1.txt"]) == 1

    def test_process_json_texts_multiple_files(self) -> None:
        """Test processing JSON texts from multiple files."""
        file_to_texts = {
            "file1.txt": {"title": "Doc1", "text": "Content 1"},
            "file2.txt": {"title": "Doc2", "text": "Content 2"},
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.TextSplitter") as mock_splitter:
            mock_instance = MagicMock()
            mock_instance.split.return_value = ["Chunk"]
            mock_splitter.return_value = mock_instance
            
            result = process_json_texts(file_to_texts, PeriodOption.NONE)
            
            assert len(result) == 2
            assert "file1.txt" in result
            assert "file2.txt" in result


class TestProcessChunks:
    def test_process_chunks_basic(self) -> None:
        """Test basic chunk processing."""
        file_to_chunks = {
            "file1.txt": [
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 1", "chunk_id": 1}),
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 2", "chunk_id": 2}),
            ]
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            result = process_chunks(file_to_chunks, 100, 1, 1)
            
            assert result.cid_to_text is not None
            assert len(result.cid_to_text) == 2
            assert result.text_to_cid is not None
            assert result.previous_cid is not None
            assert result.next_cid is not None

    def test_process_chunks_creates_previous_next_links(self) -> None:
        """Test that process_chunks creates previous/next chunk links."""
        file_to_chunks = {
            "file1.txt": [
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 1", "chunk_id": 1}),
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 2", "chunk_id": 2}),
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 3", "chunk_id": 3}),
            ]
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            result = process_chunks(file_to_chunks, 100, 1, 1)
            
            # Check previous links
            assert 2 in result.previous_cid
            assert result.previous_cid[2] == 1
            assert 3 in result.previous_cid
            assert result.previous_cid[3] == 2
            
            # Check next links
            assert 1 in result.next_cid
            assert result.next_cid[1] == 2
            assert 2 in result.next_cid
            assert result.next_cid[2] == 3

    def test_process_chunks_with_periods(self) -> None:
        """Test processing chunks with period information."""
        file_to_chunks = {
            "file1.txt": [
                json.dumps({
                    "title": "Doc1",
                    "text_chunk": "Chunk 1",
                    "chunk_id": 1,
                    "period": "2024-01"
                }),
            ]
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            result = process_chunks(file_to_chunks, 100, 1, 1)
            
            assert "ALL" in result.period_to_cids
            assert "2024-01" in result.period_to_cids

    def test_process_chunks_with_callbacks(self) -> None:
        """Test processing chunks with progress callbacks."""
        file_to_chunks = {
            "file1.txt": [
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 1", "chunk_id": 1}),
            ]
        }
        
        callback = MagicMock()
        callback.on_batch_change = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            process_chunks(file_to_chunks, 100, 1, 1, callbacks=[callback])
            
            callback.on_batch_change.assert_called()

    def test_process_chunks_empty_input(self) -> None:
        """Test processing with no chunks."""
        file_to_chunks = {}
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            result = process_chunks(file_to_chunks, 100, 1, 1)
            
            assert len(result.cid_to_text) == 0
            assert len(result.text_to_cid) == 0

    def test_process_chunks_with_invalid_json(self) -> None:
        """Test processing chunks with invalid JSON (exception path)."""
        # Create chunk string that's not valid JSON
        file_to_chunks = {
            "file1.txt": [
                "not valid json at all",
            ]
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            # Should handle gracefully without crashing
            result = process_chunks(file_to_chunks, 100, 1, 1)
            
            # Even with error, should create basic structure
            assert result.cid_to_text is not None

    def test_process_chunks_multiple_files(self) -> None:
        """Test processing chunks from multiple files."""
        file_to_chunks = {
            "file1.txt": [
                json.dumps({"title": "Doc1", "text_chunk": "Chunk 1", "chunk_id": 1}),
            ],
            "file2.txt": [
                json.dumps({"title": "Doc2", "text_chunk": "Chunk 2", "chunk_id": 1}),
            ],
        }
        
        with patch("intelligence_toolkit.query_text_data.input_processor.graph_builder") as mock_gb:
            mock_gb.update_concept_graph_edges = MagicMock()
            mock_gb.prepare_concept_graphs = MagicMock(return_value=({}, {}))
            
            result = process_chunks(file_to_chunks, 100, 1, 1)
            
            # Should have chunks from both files
            assert len(result.cid_to_text) == 2


class TestPeriodOption:
    def test_period_option_values(self) -> None:
        """Test PeriodOption enum has expected values."""
        assert PeriodOption.NONE is not None
        assert PeriodOption.DAY is not None
        assert PeriodOption.WEEK is not None
        assert PeriodOption.MONTH is not None
        assert PeriodOption.QUARTER is not None
        assert PeriodOption.YEAR is not None
