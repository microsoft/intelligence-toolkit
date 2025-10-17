# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data query_rewriter module."""

from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from intelligence_toolkit.query_text_data.query_rewriter import rewrite_query


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_rewrite_query_basic(self) -> None:
        """Test basic query rewriting."""
        ai_config = MagicMock()
        query = "What is the capital of France?"
        graph = nx.Graph()
        graph.add_edge("France", "Europe")
        graph.add_edge("capital", "city")
        graph.add_edge("Paris", "France")

        with patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.generate_text_async"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = "What is the capital city of France in Europe?"

            result = await rewrite_query(ai_config, query, graph, top_concepts=5)

            assert result == "What is the capital city of France in Europe?"
            mock_prepare.assert_called_once()
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_query_filters_dummynode(self) -> None:
        """Test that dummynode is filtered from concepts."""
        ai_config = MagicMock()
        query = "test query"
        graph = nx.Graph()
        graph.add_edge("dummynode", "concept1")
        graph.add_edge("concept2", "concept3")

        with patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.generate_text_async"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = "rewritten query"

            await rewrite_query(ai_config, query, graph, top_concepts=10)

            # Check that the concepts string doesn't contain dummynode
            call_args = mock_prepare.call_args[0]
            concepts_str = call_args[1]["concepts"]
            assert "dummynode" not in concepts_str

    @pytest.mark.asyncio
    async def test_rewrite_query_respects_top_concepts(self) -> None:
        """Test that only top N concepts are used."""
        ai_config = MagicMock()
        query = "test"
        graph = nx.Graph()
        # Create a graph with clear degree ordering
        graph.add_edges_from([("A", "B"), ("A", "C"), ("A", "D")])  # A has degree 3
        graph.add_edges_from([("B", "C")])  # B has degree 2
        # C has degree 2, D has degree 1

        with patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.generate_text_async"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = "result"

            await rewrite_query(ai_config, query, graph, top_concepts=2)

            # Verify only top 2 concepts used
            call_args = mock_prepare.call_args[0]
            concepts_str = call_args[1]["concepts"]
            # A should be included (highest degree)
            # Count commas to verify number of concepts
            concept_count = concepts_str.count(",") + 1 if concepts_str else 0
            assert concept_count <= 2

    @pytest.mark.asyncio
    async def test_rewrite_query_empty_graph(self) -> None:
        """Test rewriting with an empty concept graph."""
        ai_config = MagicMock()
        query = "test query"
        graph = nx.Graph()

        with patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.generate_text_async"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = "test query"

            result = await rewrite_query(ai_config, query, graph, top_concepts=5)

            assert result == "test query"
            # Should still be called even with empty graph
            mock_prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_query_passes_correct_params(self) -> None:
        """Test that rewrite_query passes correct parameters to utils."""
        ai_config = MagicMock()
        query = "original query"
        graph = nx.Graph()
        graph.add_edge("concept1", "concept2")

        with patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.prepare_messages"
        ) as mock_prepare, patch(
            "intelligence_toolkit.query_text_data.query_rewriter.utils.generate_text_async"
        ) as mock_generate:
            mock_prepare.return_value = [{"role": "user", "content": "test"}]
            mock_generate.return_value = "result"

            await rewrite_query(ai_config, query, graph, top_concepts=3)

            # Check prepare_messages received query
            assert mock_prepare.call_args[0][1]["query"] == "original query"
            
            # Check generate_text_async received correct stream parameter
            assert mock_generate.call_args[1]["stream"] is False
