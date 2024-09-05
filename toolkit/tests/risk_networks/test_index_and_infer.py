# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from collections import defaultdict
from unittest.mock import Mock, patch

import networkx as nx
import polars as pl
import pytest

from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.risk_networks.config import (
    SIMILARITY_THRESHOLD_MAX,
    SIMILARITY_THRESHOLD_MIN,
)
from toolkit.risk_networks.index_and_infer import (
    build_inferred_df,
    create_inferred_links,
    index_and_infer,
    index_nodes,
    infer_nodes,
)


class TestIndexNodes:
    @pytest.fixture()
    def overall_graph_small(self):
        G = nx.Graph()
        G.add_node("Entity1", type="TypeA")
        G.add_node("Entity2", type="TypeB")
        G.add_node("Entity3", type="TypeA")
        G.add_node("Entity4", type="TypeC")
        G.add_edge("Entity1", "Entity2")
        G.add_edge("Entity3", "Entity4")
        return G

    @pytest.fixture()
    def overall_graph(self):
        G = nx.Graph()
        # Adding more nodes and edges to the graph
        for i in range(1, 31):
            G.add_node(
                f"Entity{i}", type=f"Type{chr(65 + (i % 4))}"
            )  # Types will be TypeA, TypeB, TypeC
        for i in range(1, 31, 2):
            G.add_edge(f"Entity{i}", f"Entity{i + 1}")
        return G

    async def test_index_nodes_empty_types(self, overall_graph):
        with pytest.raises(
            ValueError,
            match="No node types to index",
        ):
            await index_nodes([], overall_graph)

    @patch("toolkit.risk_networks.index_and_infer.OpenAIEmbedder")
    async def test_index_nodes_small_samples(self, mock_embedder, overall_graph_small):
        async def embed_store_many(*args) -> list[list[float]]:
            return [
                {"vector": [0.1, 0.3], "text": "A"},
                {"vector": [0.3, 0.4], "text": "B"},
            ]

        mock_instance = mock_embedder.return_value
        mock_instance.embed_store_many.side_effect = embed_store_many

        indexed_node_types = ["TypeA", "TypeB"]
        # Expect ValueError
        with pytest.raises(
            ValueError,
            match="Expected n_neighbors <= n_samples_fit, but n_neighbors = 20, n_samples_fit = 2, n_samples = 2",
        ):
            await index_nodes(indexed_node_types, overall_graph_small)

    @patch("toolkit.risk_networks.index_and_infer.OpenAIEmbedder")
    async def test_index_nodes(self, mock_embedder, overall_graph):
        async def embed_store_many(*args) -> list[list[float]]:
            return [
                {"vector": [0.1, 0.3], "text": "A"},
            ] * 23

        mock_instance = mock_embedder.return_value
        mock_instance.embed_store_many.side_effect = embed_store_many

        indexed_node_types = ["TypeA", "TypeB", "TypeC"]
        (
            embedded_texts,
            nearest_text_distances,
            nearest_text_indices,
        ) = await index_nodes(indexed_node_types, overall_graph)

        # Check if the embedded texts are correct
        expected_texts = [
            f"Entity{i}"
            for i in range(1, 31)
            if f"Type{chr(65 + (i % 4))}" in indexed_node_types
        ]
        expected_texts.sort()

        # Check the shape of the distances and indices
        expected_shape = (len(expected_texts), 20)
        assert embedded_texts == expected_texts
        assert nearest_text_distances.shape == expected_shape
        assert nearest_text_indices.shape == expected_shape


class TestInferNodes:
    def test_infer_nodes_min_threshold(self):
        with pytest.raises(
            ValueError,
            match=f"Similarity threshold must be between {SIMILARITY_THRESHOLD_MIN} and {SIMILARITY_THRESHOLD_MAX}",
        ):
            infer_nodes(-0.1, [], [], [])

    def test_infer_nodes_max_threshold(self):
        with pytest.raises(
            ValueError,
            match=f"Similarity threshold must be between {SIMILARITY_THRESHOLD_MIN} and {SIMILARITY_THRESHOLD_MAX}",
        ):
            infer_nodes(2, [], [], [])

    def test_infer_nodes_005(self):
        similarity_threshold = 0.05
        embedded_texts = [
            "Entity==ABCDE",
            "Entity==PLUS_ONE",
            "Entity==PLUS ONE",
        ]
        nearest_text_indices = [[1, 2], [2, 1], [1, 2]]
        nearest_text_distances = [
            [0.1, 0.1, 0.3],
            [0.02, 0.1, 0.3],
            [0.02, 0.1, 0.3],
        ]

        inferred_links = infer_nodes(
            similarity_threshold,
            embedded_texts,
            nearest_text_indices,
            nearest_text_distances,
        )

        expected_links = defaultdict(set)
        expected_links["Entity==PLUS ONE"].add("Entity==PLUS_ONE")
        expected_links["Entity==PLUS_ONE"].add("Entity==PLUS ONE")

        assert inferred_links == expected_links

    def test_infer_nodes_1(self):
        similarity_threshold = 1
        embedded_texts = [
            "Entity==ABCDE",
            "Entity==PLUS_ONE",
            "Entity==PLUS ONE",
        ]
        nearest_text_indices = [[1, 2], [2, 1], [1, 2]]
        nearest_text_distances = [
            [0.1, 0.1, 0.3],
            [0.02, 0.1, 0.3],
            [0.02, 0.1, 0.3],
        ]

        inferred_links = infer_nodes(
            similarity_threshold,
            embedded_texts,
            nearest_text_indices,
            nearest_text_distances,
        )

        expected_links = defaultdict(set)
        expected_links["Entity==PLUS ONE"].add("Entity==PLUS_ONE")
        expected_links["Entity==PLUS ONE"].add("Entity==ABCDE")
        expected_links["Entity==PLUS_ONE"].add("Entity==PLUS ONE")
        expected_links["Entity==PLUS_ONE"].add("Entity==ABCDE")
        expected_links["Entity==ABCDE"].add("Entity==PLUS ONE")
        expected_links["Entity==ABCDE"].add("Entity==PLUS_ONE")

        assert inferred_links == expected_links

    def test_infer_nodes_07(self):
        similarity_threshold = 0.7
        embedded_texts = [
            "Entity==ABCDE",
            "Entity==PLUS_ONE",
            "Entity==PLUS ONE",
        ]
        nearest_text_indices = [[1, 2], [2, 1], [1, 2]]
        nearest_text_distances = [
            [0.8, 0.1, 0.3],
            [0.02, 0.1, 0.3],
            [0.02, 0.1, 0.3],
        ]

        inferred_links = infer_nodes(
            similarity_threshold,
            embedded_texts,
            nearest_text_indices,
            nearest_text_distances,
        )

        expected_links = defaultdict(set)
        expected_links["Entity==PLUS ONE"].add("Entity==PLUS_ONE")
        expected_links["Entity==PLUS ONE"].add("Entity==ABCDE")
        expected_links["Entity==PLUS_ONE"].add("Entity==PLUS ONE")
        expected_links["Entity==ABCDE"].add("Entity==PLUS ONE")

        assert inferred_links == expected_links

    def test_infer_nodes_progress_callbacks_empty(self):
        similarity_threshold = 0.7
        embedded_texts = [
            "Entity==ABCDE",
            "Entity==PLUS_ONE",
            "Entity==PLUS ONE",
        ]
        nearest_text_indices = [[1, 2], [2, 1], [1, 2]]
        nearest_text_distances = [
            [0.8, 0.1, 0.3],
            [0.02, 0.1, 0.3],
            [0.02, 0.1, 0.3],
        ]

        inferred_links = infer_nodes(
            similarity_threshold,
            embedded_texts,
            nearest_text_indices,
            nearest_text_distances,
            progress_callbacks=[],
        )

        expected_links = defaultdict(set)
        expected_links["Entity==PLUS ONE"].add("Entity==PLUS_ONE")
        expected_links["Entity==PLUS ONE"].add("Entity==ABCDE")
        expected_links["Entity==PLUS_ONE"].add("Entity==PLUS ONE")
        expected_links["Entity==ABCDE"].add("Entity==PLUS ONE")

        assert inferred_links == expected_links

    def test_infer_nodes_one_progress_callback(self):
        similarity_threshold = 0.7
        embedded_texts = [
            "Entity==ABCDE",
            "Entity==PLUS_ONE",
            "Entity==PLUS ONE",
        ]
        nearest_text_indices = [[1, 2], [2, 1], [1, 2]]
        nearest_text_distances = [
            [0.8, 0.1, 0.3],
            [0.02, 0.1, 0.3],
            [0.02, 0.1, 0.3],
        ]

        callb1 = ProgressBatchCallback()
        progress_callback = Mock()
        callb1.on_batch_change = progress_callback

        infer_nodes(
            similarity_threshold,
            embedded_texts,
            nearest_text_indices,
            nearest_text_distances,
            progress_callbacks=[callb1],
        )
        progress_callback.assert_called_with(2, 3, "Infering links...")

    def test_infer_nodes_two_progress_callback(self):
        similarity_threshold = 0.7
        embedded_texts = [
            "Entity==ABCDE",
            "Entity==PLUS_ONE",
            "Entity==PLUS ONE",
        ]
        nearest_text_indices = [[1, 2], [2, 1], [1, 2]]
        nearest_text_distances = [
            [0.8, 0.1, 0.3],
            [0.02, 0.1, 0.3],
            [0.02, 0.1, 0.3],
        ]

        callb1 = ProgressBatchCallback()
        progress_callback = Mock()
        callb1.on_batch_change = progress_callback

        callb2 = ProgressBatchCallback()
        progress_callback2 = Mock()
        callb2.on_batch_change = progress_callback2

        infer_nodes(
            similarity_threshold,
            embedded_texts,
            nearest_text_indices,
            nearest_text_distances,
            progress_callbacks=[callb1, callb2],
        )
        progress_callback.assert_called_with(2, 3, "Infering links...")
        progress_callback2.assert_called_with(2, 3, "Infering links...")


class TestCreateInferredLinks:
    def test_create_links(self):
        inferred_links = defaultdict(set)
        inferred_links["Entity==PLUS ONE"].add("Entity==PLUS_ONE")
        inferred_links["Entity==PLUS ONE"].add("Entity==ABCDE")
        inferred_links["Entity==PLUS_ONE"].add("Entity==PLUS ONE")
        inferred_links["Entity==ABCDE"].add("Entity==PLUS ONE")

        created_links = create_inferred_links(inferred_links)

        expected_links = [
            ("Entity==PLUS ONE", "Entity==PLUS_ONE"),
            ("Entity==ABCDE", "Entity==PLUS ONE"),
        ]
        assert created_links == expected_links

    def test_create_links_return_empty(self):
        inferred_links = defaultdict(set)

        created_links = create_inferred_links(inferred_links)

        assert created_links == []

        created_links = create_inferred_links(inferred_links)

        assert created_links == []


class TestBuildInferredDF:
    def test_build_inferred_entity_df(self) -> None:
        inferred_links = defaultdict(set)
        inferred_links["ENTITY==ABCDE"].add("ENTITY==ABCDEF")
        inferred_links["ENTITY==ABCDEFGR"].add("ENTITY==ABCDEFGRA")
        inferred_links["ENTITY==PLUS ONE"].add("ENTITY==PLUS_ONE")

        inferred_df = build_inferred_df(inferred_links)

        expected_df = pl.DataFrame(
            {
                "text": ["ABCDE", "PLUS ONE", "ABCDEFGR"],
                "similar": ["ABCDEF", "PLUS_ONE", "ABCDEFGRA"],
            }
        ).sort(["text", "similar"])

        assert inferred_df.equals(expected_df)

    def test_build_inferred_attribute_df(self) -> None:
        inferred_links = defaultdict(set)
        inferred_links["attr1==ABCDE"].add("attr1==ABCDEF")
        inferred_links["attr1==PLUS ONE"].add("attr1==PLUS_ONE")
        inferred_links["attr1==ABCDEF F"].add("attr1==ABCDEFF_FA")

        inferred_df = build_inferred_df(inferred_links)

        expected_df = pl.DataFrame(
            {
                "text": ["attr1==ABCDE", "attr1==ABCDEF F", "attr1==PLUS ONE"],
                "similar": ["attr1==ABCDEF", "attr1==ABCDEFF_FA", "attr1==PLUS_ONE"],
            }
        ).sort(["text", "similar"])

        assert inferred_df.equals(expected_df)

    def test_build_inferred_df_empty(self) -> None:
        inferred_links = defaultdict(set)
        inferred_df = build_inferred_df(inferred_links)

        expected_df = pl.DataFrame(
            {
                "text": [],
                "similar": [],
            }
        )

        assert inferred_df.equals(expected_df)


class TestIndexAndInfer:
    @pytest.fixture()
    def overall_graph(self):
        G = nx.Graph()
        # Adding more nodes and edges to the graph
        for i in range(1, 31):
            G.add_node(f"Entity{i}", type=f"Type{chr(65 + (i % 4))}")
        for i in range(1, 31, 2):
            G.add_edge(f"Entity{i}", f"Entity{i + 1}")
        return G

    async def test_empty_graph(self):
        indexed_node_types = ["TypeA", "TypeB", "TypeC"]
        with pytest.raises(ValueError, match="Graph is empty"):
            await index_and_infer(indexed_node_types, nx.Graph(), 0)

    @patch("toolkit.risk_networks.index_and_infer.index_nodes")
    @patch("toolkit.risk_networks.index_and_infer.infer_nodes")
    async def test_index_and_infer(
        self, mock_infer_nodes, mock_index_nodes, overall_graph
    ) -> None:
        indexed_node_types = ["TypeA", "TypeB", "TypeC"]
        embedded_texts = ["Entity1", "Entity2", "Entity3"]
        nearest_text_distances = [[0.1, 0.3], [0.3, 0.4]]
        nearest_text_indices = [[0, 1], [1, 0]]

        mock_index_nodes.return_value = (
            embedded_texts,
            nearest_text_distances,
            nearest_text_indices,
        )

        inferred_links = defaultdict(set)
        inferred_links["Entity1"].add("Entity2")
        mock_infer_nodes.return_value = inferred_links

        link_list, _ = await index_and_infer(indexed_node_types, overall_graph, 0.5)

        assert link_list == inferred_links

        mock_index_nodes.assert_called_once_with(
            indexed_node_types, overall_graph, None, None, None, True
        )
        mock_infer_nodes.assert_called_once_with(
            0.5, embedded_texts, nearest_text_indices, nearest_text_distances, None
        )
