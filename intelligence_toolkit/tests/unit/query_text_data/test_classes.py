# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data classes module."""

from unittest.mock import MagicMock

import networkx as nx

from intelligence_toolkit.query_text_data.classes import (
    AnswerObject,
    ChunkSearchConfig,
    ProcessedChunks,
)


class TestProcessedChunks:
    def test_initialization(self) -> None:
        """Test ProcessedChunks initialization."""
        cid_to_text = {1: "text1", 2: "text2"}
        text_to_cid = {"text1": 1, "text2": 2}
        period_graphs = {"period1": nx.Graph()}
        hierarchical = MagicMock()
        community_label = {0: {1: "label1"}}
        concept_to_cids = {"concept1": [1, 2]}
        cid_to_concepts = {1: ["concept1"], 2: ["concept1"]}
        previous_cid = {2: 1}
        next_cid = {1: 2}
        period_to_cids = {"period1": [1, 2]}
        node_counts = {"concept1": {"period1": 2}}
        edge_counts = {("c1", "c2"): {"period1": 1}}

        chunks = ProcessedChunks(
            cid_to_text=cid_to_text,
            text_to_cid=text_to_cid,
            period_concept_graphs=period_graphs,
            hierarchical_communities=hierarchical,
            community_to_label=community_label,
            concept_to_cids=concept_to_cids,
            cid_to_concepts=cid_to_concepts,
            previous_cid=previous_cid,
            next_cid=next_cid,
            period_to_cids=period_to_cids,
            node_period_counts=node_counts,
            edge_period_counts=edge_counts,
        )

        assert chunks.cid_to_text == cid_to_text
        assert chunks.text_to_cid == text_to_cid
        assert chunks.period_concept_graphs == period_graphs
        assert chunks.hierarchical_communities == hierarchical
        assert chunks.community_to_label == community_label
        assert chunks.concept_to_cids == concept_to_cids
        assert chunks.cid_to_concepts == cid_to_concepts
        assert chunks.previous_cid == previous_cid
        assert chunks.next_cid == next_cid
        assert chunks.period_to_cids == period_to_cids
        assert chunks.node_period_counts == node_counts
        assert chunks.edge_period_counts == edge_counts

    def test_repr(self) -> None:
        """Test ProcessedChunks __repr__."""
        chunks = ProcessedChunks(
            cid_to_text={1: "text1", 2: "text2", 3: "text3"},
            text_to_cid={},
            period_concept_graphs={},
            hierarchical_communities=MagicMock(),
            community_to_label={},
            concept_to_cids={},
            cid_to_concepts={},
            previous_cid={},
            next_cid={},
            period_to_cids={},
            node_period_counts={},
            edge_period_counts={},
        )

        repr_str = repr(chunks)
        assert "ProcessedChunks" in repr_str
        assert "num_chunks=3" in repr_str


class TestChunkSearchConfig:
    def test_initialization(self) -> None:
        """Test ChunkSearchConfig initialization."""
        config = ChunkSearchConfig(
            adjacent_test_steps=2,
            community_relevance_tests=5,
            community_ranking_chunks=10,
            relevance_test_batch_size=20,
            relevance_test_budget=100,
            irrelevant_community_restart=3,
            analysis_update_interval=10,
        )

        assert config.adjacent_test_steps == 2
        assert config.community_relevance_tests == 5
        assert config.community_ranking_chunks == 10
        assert config.relevance_test_batch_size == 20
        assert config.relevance_test_budget == 100
        assert config.irrelevant_community_restart == 3
        assert config.analysis_update_interval == 10

    def test_initialization_with_defaults(self) -> None:
        """Test ChunkSearchConfig initialization with default analysis_update_interval."""
        config = ChunkSearchConfig(
            adjacent_test_steps=2,
            community_relevance_tests=5,
            community_ranking_chunks=10,
            relevance_test_batch_size=20,
            relevance_test_budget=100,
            irrelevant_community_restart=3,
        )

        assert config.analysis_update_interval == 0

    def test_repr(self) -> None:
        """Test ChunkSearchConfig __repr__."""
        config = ChunkSearchConfig(
            adjacent_test_steps=2,
            community_relevance_tests=5,
            community_ranking_chunks=10,
            relevance_test_batch_size=20,
            relevance_test_budget=100,
            irrelevant_community_restart=3,
        )

        repr_str = repr(config)
        assert "ChunkSearchConfig" in repr_str
        assert "adjacent_test_steps=2" in repr_str
        assert "community_relevance_tests=5" in repr_str
        assert "relevance_test_batch_size=20" in repr_str


class TestAnswerObject:
    def test_initialization(self) -> None:
        """Test AnswerObject initialization."""
        answer = AnswerObject(
            extended_answer="This is the answer to your question.",
            references=["source1", "source2"],
            referenced_chunks=[1, 2, 3],
            net_new_sources=2,
        )

        assert answer.extended_answer == "This is the answer to your question."
        assert answer.references == ["source1", "source2"]
        assert answer.referenced_chunks == [1, 2, 3]
        assert answer.net_new_sources == 2

    def test_repr(self) -> None:
        """Test AnswerObject __repr__."""
        long_answer = "This is a very long answer " * 20  # Make it > 100 chars
        answer = AnswerObject(
            extended_answer=long_answer,
            references=["source1", "source2", "source3"],
            referenced_chunks=[1, 2, 3, 4, 5],
            net_new_sources=3,
        )

        repr_str = repr(answer)
        assert "AnswerObject" in repr_str
        assert "references=3" in repr_str
        assert "referenced_chunks=5" in repr_str
        assert "net_new_sources=3" in repr_str
        # Check that answer is truncated in repr
        assert len(long_answer) > 100
        assert long_answer[:100] in repr_str

    def test_empty_lists(self) -> None:
        """Test AnswerObject with empty lists."""
        answer = AnswerObject(
            extended_answer="Answer",
            references=[],
            referenced_chunks=[],
            net_new_sources=0,
        )

        assert answer.references == []
        assert answer.referenced_chunks == []
        assert answer.net_new_sources == 0
