# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data api module."""

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from intelligence_toolkit.query_text_data.api import QueryTextData, QueryTextDataStage
from intelligence_toolkit.query_text_data.classes import ChunkSearchConfig, ProcessedChunks


class TestQueryTextDataInit:
    def test_init(self) -> None:
        """Test QueryTextData initialization."""
        qtd = QueryTextData()
        
        assert qtd.stage == QueryTextDataStage.INITIAL
        assert qtd.label_to_chunks is None
        assert qtd.processed_chunks is None
        assert qtd.cid_to_vector is None

    def test_repr(self) -> None:
        """Test QueryTextData string representation."""
        qtd = QueryTextData()
        assert repr(qtd) == "QueryTextData()"


class TestSetAiConfig:
    def test_set_ai_config(self) -> None:
        """Test setting AI configuration."""
        qtd = QueryTextData()
        ai_config = {"model": "gpt-4"}
        cache = "test_cache"
        
        qtd.set_ai_config(ai_config, cache)
        
        assert qtd.ai_configuration == ai_config
        assert qtd.embedding_cache == cache


class TestResetWorkflow:
    def test_reset_workflow(self) -> None:
        """Test resetting workflow to initial state."""
        qtd = QueryTextData()
        
        # Set some values
        qtd.stage = QueryTextDataStage.CHUNKS_CREATED
        qtd.label_to_chunks = {"file1": ["chunk1"]}
        qtd.query = "test query"
        
        # Reset
        qtd.reset_workflow()
        
        assert qtd.stage == QueryTextDataStage.INITIAL
        assert qtd.label_to_chunks is None
        assert qtd.query is None
        assert qtd.expanded_query is None
        assert qtd.chunk_search_config is None
        assert qtd.relevant_cids is None
        assert qtd.search_summary is None
        assert qtd.answer_config is None
        assert qtd.answer_object is None
        assert qtd.level_to_label_to_network is None


class TestSetEmbedder:
    def test_set_embedder(self) -> None:
        """Test setting text embedder."""
        qtd = QueryTextData()
        embedder = MagicMock()
        
        qtd.set_embedder(embedder)
        
        assert qtd.text_embedder == embedder


class TestProcessDataFromFiles:
    def test_process_data_from_files(self) -> None:
        """Test processing data from files."""
        qtd = QueryTextData()
        
        with patch("intelligence_toolkit.query_text_data.api.document_processor") as mock_dp:
            mock_dp.convert_files_to_chunks.return_value = {"file1.txt": ["chunk1", "chunk2"]}
            
            result = qtd.process_data_from_files(["file1.txt"], chunk_size=1000)
            
            assert result == {"file1.txt": ["chunk1", "chunk2"]}
            assert qtd.label_to_chunks == {"file1.txt": ["chunk1", "chunk2"]}
            assert qtd.stage == QueryTextDataStage.CHUNKS_CREATED

    def test_process_data_from_files_with_callbacks(self) -> None:
        """Test processing data from files with callbacks."""
        qtd = QueryTextData()
        callback = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.api.document_processor") as mock_dp:
            mock_dp.convert_files_to_chunks.return_value = {"file1.txt": ["chunk1"]}
            
            qtd.process_data_from_files(["file1.txt"], callbacks=[callback])
            
            mock_dp.convert_files_to_chunks.assert_called_once()


class TestProcessTextChunks:
    def test_process_text_chunks(self) -> None:
        """Test processing text chunks."""
        qtd = QueryTextData()
        qtd.label_to_chunks = {"file1": ["chunk1"]}
        
        mock_processed = MagicMock(spec=ProcessedChunks)
        
        with patch("intelligence_toolkit.query_text_data.api.input_processor") as mock_ip:
            mock_ip.process_chunks.return_value = mock_processed
            
            result = qtd.process_text_chunks(max_cluster_size=25)
            
            assert result == mock_processed
            assert qtd.processed_chunks == mock_processed
            assert qtd.stage == QueryTextDataStage.CHUNKS_PROCESSED

    def test_process_text_chunks_with_parameters(self) -> None:
        """Test processing text chunks with custom parameters."""
        qtd = QueryTextData()
        qtd.label_to_chunks = {"file1": ["chunk1"]}
        
        with patch("intelligence_toolkit.query_text_data.api.input_processor") as mock_ip:
            mock_ip.process_chunks.return_value = MagicMock()
            
            qtd.process_text_chunks(
                max_cluster_size=50,
                min_edge_weight=3,
                min_node_degree=3
            )
            
            mock_ip.process_chunks.assert_called_once_with(
                qtd.label_to_chunks,
                50,
                3,
                3,
                callbacks=[]
            )


class TestEmbedTextChunks:
    @pytest.mark.asyncio
    async def test_embed_text_chunks(self) -> None:
        """Test embedding text chunks."""
        qtd = QueryTextData()
        qtd.processed_chunks = MagicMock()
        qtd.processed_chunks.cid_to_text = {1: "text1", 2: "text2"}
        qtd.text_embedder = MagicMock()
        qtd.embedding_cache = "cache"
        
        mock_vectors = {1: [0.1, 0.2], 2: [0.3, 0.4]}
        
        with patch("intelligence_toolkit.query_text_data.api.helper_functions") as mock_hf:
            mock_hf.embed_texts = AsyncMock(return_value=mock_vectors)
            
            result = await qtd.embed_text_chunks()
            
            assert result == mock_vectors
            assert qtd.cid_to_vector == mock_vectors
            assert qtd.stage == QueryTextDataStage.CHUNKS_EMBEDDED


class TestAnchorQueryToConcepts:
    @pytest.mark.asyncio
    async def test_anchor_query_to_concepts(self) -> None:
        """Test anchoring query to concepts."""
        qtd = QueryTextData()
        qtd.ai_configuration = {"model": "gpt-4"}
        qtd.processed_chunks = MagicMock()
        qtd.processed_chunks.period_concept_graphs = {"ALL": MagicMock()}
        
        with patch("intelligence_toolkit.query_text_data.api.query_rewriter") as mock_qr:
            mock_qr.rewrite_query = AsyncMock(return_value="anchored query")
            
            result = await qtd.anchor_query_to_concepts("test query", top_concepts=50)
            
            assert result == "anchored query"
            mock_qr.rewrite_query.assert_called_once()


class TestDetectRelevantTextChunks:
    @pytest.mark.asyncio
    async def test_detect_relevant_text_chunks(self) -> None:
        """Test detecting relevant text chunks."""
        qtd = QueryTextData()
        qtd.ai_configuration = {"model": "gpt-4"}
        qtd.processed_chunks = MagicMock()
        qtd.processed_chunks.cid_to_text = {1: "text1"}
        qtd.cid_to_vector = {1: [0.1, 0.2]}
        qtd.text_embedder = MagicMock()
        qtd.embedding_cache = "cache"
        
        chunk_config = ChunkSearchConfig(
            adjacent_test_steps=1,
            community_relevance_tests=5,
            community_ranking_chunks=10,
            relevance_test_batch_size=10,
            relevance_test_budget=100,
            irrelevant_community_restart=3
        )
        
        with patch("intelligence_toolkit.query_text_data.api.relevance_assessor") as mock_ra:
            mock_ra.detect_relevant_chunks = AsyncMock(return_value=([1, 2], "summary"))
            
            result = await qtd.detect_relevant_text_chunks(
                "query",
                "expanded query",
                chunk_config
            )
            
            assert result == ([1, 2], "summary")
            assert qtd.relevant_cids == [1, 2]
            assert qtd.search_summary == "summary"
            assert qtd.stage == QueryTextDataStage.CHUNKS_MINED

    @pytest.mark.asyncio
    async def test_detect_relevant_text_chunks_with_callbacks(self) -> None:
        """Test detecting relevant text chunks with callbacks."""
        qtd = QueryTextData()
        qtd.ai_configuration = {"model": "gpt-4"}
        qtd.processed_chunks = MagicMock()
        qtd.processed_chunks.cid_to_text = {1: "text1"}
        qtd.cid_to_vector = {1: [0.1, 0.2]}
        qtd.text_embedder = MagicMock()
        qtd.embedding_cache = "cache"
        
        chunk_config = ChunkSearchConfig(
            adjacent_test_steps=1,
            community_relevance_tests=5,
            community_ranking_chunks=10,
            relevance_test_batch_size=10,
            relevance_test_budget=100,
            irrelevant_community_restart=3
        )
        chunk_callback = MagicMock()
        analysis_callback = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.api.relevance_assessor") as mock_ra:
            mock_ra.detect_relevant_chunks = AsyncMock(return_value=([1], "summary"))
            
            await qtd.detect_relevant_text_chunks(
                "query",
                "expanded",
                chunk_config,
                chunk_callback=chunk_callback,
                analysis_callback=analysis_callback
            )
            
            # Should have created commentary with callbacks
            assert qtd.commentary is not None


class TestAnswerQueryWithRelevantChunks:
    @pytest.mark.asyncio
    async def test_answer_query_with_relevant_chunks(self) -> None:
        """Test answering query with relevant chunks."""
        qtd = QueryTextData()
        qtd.ai_configuration = {"model": "gpt-4"}
        qtd.query = "test query"
        qtd.expanded_query = "expanded query"
        qtd.processed_chunks = MagicMock()
        qtd.commentary = MagicMock()
        
        mock_answer = MagicMock()
        
        with patch("intelligence_toolkit.query_text_data.api.answer_builder") as mock_ab:
            mock_ab.answer_query = AsyncMock(return_value=mock_answer)
            
            result = await qtd.answer_query_with_relevant_chunks()
            
            assert result == mock_answer
            assert qtd.answer_object == mock_answer
            assert qtd.stage == QueryTextDataStage.QUESTION_ANSWERED


class TestBuildConceptCommunityGraph:
    def test_build_concept_community_graph(self) -> None:
        """Test building concept community graph."""
        qtd = QueryTextData()
        qtd.processed_chunks = MagicMock()
        qtd.processed_chunks.period_concept_graphs = {"ALL": MagicMock()}
        qtd.processed_chunks.hierarchical_communities = []
        
        mock_graph = {"level0": {"comm1": MagicMock()}}
        
        with patch("intelligence_toolkit.query_text_data.api.graph_builder") as mock_gb:
            mock_gb.build_meta_graph.return_value = mock_graph
            
            result = qtd.build_concept_community_graph()
            
            assert result == mock_graph
            assert qtd.level_to_label_to_network == mock_graph


class TestCondenseAnswer:
    def test_condense_answer(self) -> None:
        """Test condensing answer."""
        qtd = QueryTextData()
        qtd.ai_configuration = {"model": "gpt-4"}
        qtd.query = "test query"
        qtd.answer_object = MagicMock()
        qtd.answer_object.extended_answer = "long answer"
        
        with patch("intelligence_toolkit.query_text_data.api.utils") as mock_utils:
            with patch("intelligence_toolkit.query_text_data.api.OpenAIClient") as mock_client_class:
                mock_utils.generate_messages.return_value = [{"role": "user", "content": "test"}]
                mock_client = MagicMock()
                mock_client.generate_chat.return_value = "condensed answer"
                mock_client_class.return_value = mock_client
                
                result = qtd.condense_answer()
                
                assert result == "condensed answer"
                assert qtd.condensed_answer == "condensed answer"


class TestPrepareForNewQuery:
    def test_prepare_for_new_query(self) -> None:
        """Test preparing for new query."""
        qtd = QueryTextData()
        
        # Set some query-specific values
        qtd.query = "test query"
        qtd.expanded_query = "expanded"
        qtd.relevant_cids = [1, 2, 3]
        qtd.answer_object = MagicMock()
        qtd.stage = QueryTextDataStage.QUESTION_ANSWERED
        
        qtd.prepare_for_new_query()
        
        assert qtd.query is None
        assert qtd.expanded_query is None
        assert qtd.relevant_cids is None
        assert qtd.answer_object is None
        assert qtd.stage == QueryTextDataStage.CHUNKS_EMBEDDED


class TestPrepareForNewAnswer:
    def test_prepare_for_new_answer(self) -> None:
        """Test preparing for new answer."""
        qtd = QueryTextData()
        
        qtd.answer_config = {"config": "value"}
        qtd.answer_object = MagicMock()
        qtd.stage = QueryTextDataStage.QUESTION_ANSWERED
        
        qtd.prepare_for_new_answer()
        
        assert qtd.answer_config is None
        assert qtd.answer_object is None
        assert qtd.stage == QueryTextDataStage.CHUNKS_MINED


class TestGetChunksAsDf:
    def test_get_chunks_as_df(self) -> None:
        """Test getting chunks as DataFrame."""
        qtd = QueryTextData()
        qtd.label_to_chunks = {
            "file1.txt": [
                json.dumps({"text": "chunk1"}),
                json.dumps({"text": "chunk2"})
            ],
            "file2.txt": [
                json.dumps({"text": "chunk3"})
            ]
        }
        
        result = qtd.get_chunks_as_df()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "file_name" in result.columns
        assert "text_to_label_str" in result.columns
        assert result.iloc[0]["file_name"] == "file1.txt"


class TestImportChunksFromStr:
    def test_import_chunks_from_str(self) -> None:
        """Test importing chunks from string."""
        qtd = QueryTextData()
        
        csv_data = StringIO("""file_name,text_to_label_str
file1.txt,{"text": "chunk1"}
file1.txt,{"text": "chunk2"}
file2.txt,{"text": "chunk3"}""")
        
        qtd.import_chunks_from_str(csv_data)
        
        assert "file1.txt" in qtd.label_to_chunks
        assert "file2.txt" in qtd.label_to_chunks
        assert len(qtd.label_to_chunks["file1.txt"]) == 2
        assert len(qtd.label_to_chunks["file2.txt"]) == 1


class TestGenerateAnalysisCommentary:
    @pytest.mark.asyncio
    async def test_generate_analysis_commentary(self) -> None:
        """Test generating analysis commentary."""
        qtd = QueryTextData()
        qtd.commentary = MagicMock()
        qtd.commentary.generate_commentary = AsyncMock(return_value="commentary text")
        
        result = await qtd.generate_analysis_commentary()
        
        assert result == "commentary text"
        qtd.commentary.generate_commentary.assert_called_once()


class TestQueryTextDataStage:
    def test_stage_enum_values(self) -> None:
        """Test QueryTextDataStage enum values."""
        assert QueryTextDataStage.INITIAL.value == 0
        assert QueryTextDataStage.CHUNKS_CREATED.value == 1
        assert QueryTextDataStage.CHUNKS_PROCESSED.value == 2
        assert QueryTextDataStage.CHUNKS_EMBEDDED.value == 3
        assert QueryTextDataStage.CHUNKS_MINED.value == 4
        assert QueryTextDataStage.QUESTION_ANSWERED.value == 5
