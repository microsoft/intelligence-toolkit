from collections import defaultdict
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import pytest

from python.helpers.progress_batch_callback import ProgressBatchCallback
from python.risk_networks.constants import (
    SIMILARITY_THRESHOLD_MAX,
    SIMILARITY_THRESHOLD_MIN,
)
from python.risk_networks.graph_functions import (
    _merge_condition,
    _merge_node_list,
    _merge_nodes,
    build_undirected_graph,
    create_links,
    index_nodes,
    infer_nodes,
    simplify_graph,
)


@pytest.fixture()
def graph():
    G = nx.Graph()
    G.add_node("A", type="TypeA", flags=1)

    G.add_node("B", type="TypeB", flags=0)
    G.add_node("C", type="TypeC", flags=1)
    G.add_node("D", type="TypeC", flags=1)
    G.add_node("E", type="TypeC", flags=0)
    G.add_node("F", type="TypeC", flags=1)
    G.add_edge("A", "B")
    G.add_edge("B", "F")
    G.add_edge("B", "C")
    G.add_edge("E", "F")
    G.add_edge("D", "F")
    G.add_edge("A", "F")
    return G


def test_merge_node_list(graph):
    merge_list = ["A", "C"]
    merged_graph = _merge_node_list(graph, merge_list)

    assert merged_graph.has_node("A;C")
    assert merged_graph.has_node("B")
    assert merged_graph.nodes["A;C"]["type"] == "TypeA;TypeC"
    assert merged_graph.nodes["A;C"]["flags"] == 1
    assert merged_graph.has_edge("B", "A;C")
    assert not merged_graph.has_node("A")
    assert not merged_graph.has_node("C")
    assert not merged_graph.has_edge("A", "B")
    assert not merged_graph.has_edge("B", "C")
    assert not merged_graph.has_edge("A", "B")
    assert not merged_graph.has_edge("B", "C")
    assert not merged_graph.has_edge("B", "C")


def test_merge_condition():
    x = "A==1;B==2;C==3"
    y = "D==4;E==1;F==5"
    assert _merge_condition(x, y) is True

    # Test case 2: x and y have a common attribute value in the second position
    x = "A==1;B==2;C==3"
    y = "D==4;E==5;F==2"
    assert _merge_condition(x, y) is True

    # Test case 3: x and y do not have any common attribute values
    x = "A==1;B==2;C==3"
    y = "D==4;E==5;F==6"
    assert _merge_condition(x, y) is False

    # Test case 4: x and y have multiple common attribute values
    x = "A==1;B==2;C==3"
    y = "D==4;E==1;F==2"
    assert _merge_condition(x, y) is True


def merge_node_list(G, merge_list):
    # Mock implementation: merges nodes by combining them into a single node.
    # This is a placeholder for the actual merge logic.
    G.remove_nodes_from(merge_list[1:])
    G.add_node(merge_list[0])
    return G


def test_merge_all_nodes(graph):
    merged_graph = _merge_nodes(graph, lambda _x, _y: True)
    assert len(merged_graph.nodes()) < len(graph.nodes())


def test_merge_no_nodes(graph):
    merged_graph = _merge_nodes(graph, lambda _x, _y: False)
    assert len(merged_graph.nodes()) == len(graph.nodes())


def test_empty_graph():
    G = nx.Graph()
    merged_graph = _merge_nodes(G, lambda _x, _y: True)
    assert len(merged_graph.nodes()) == 0


def test_simplify_condition_false(mocker, graph):
    mocker.patch(
        "python.risk_networks.graph_functions._merge_nodes"
    ).return_value = graph

    aba = simplify_graph(graph)
    assert len(aba.nodes()) == 3


def test_simplify_condition_true(mocker, graph):
    G = nx.Graph()
    mocker.patch("python.risk_networks.graph_functions._merge_nodes").return_value = G

    aba = simplify_graph(graph)
    assert len(aba.nodes()) == 0


class TestBuildUndirectedGraph:
    def test_build_undirected_graph_empty(self):
        result = build_undirected_graph()
        assert len(result.nodes()) == 0
        assert len(result.edges()) == 0

    def test_network_attribute_links(self):
        network_attribute_links = [
            [("Entity1", "attribute", "Value1"), ("Entity2", "attribute", "Value2")],
            [("Entity3", "relation", "Entity4"), ("Entity5", "relation", "Entity6")],
            [("Entity7", "attribute", "Value3")],
        ]
        result = build_undirected_graph(network_attribute_links)
        expected_nodes = [
            "ENTITY==Entity1",
            "attribute==Value1",
            "ENTITY==Entity2",
            "attribute==Value2",
            "ENTITY==Entity3",
            "relation==Entity4",
            "ENTITY==Entity5",
            "relation==Entity6",
            "ENTITY==Entity7",
            "attribute==Value3",
        ]

        expected_edges = [
            ("ENTITY==Entity1", "attribute==Value1"),
            ("ENTITY==Entity2", "attribute==Value2"),
            ("ENTITY==Entity3", "relation==Entity4"),
            ("ENTITY==Entity5", "relation==Entity6"),
            ("ENTITY==Entity7", "attribute==Value3"),
        ]
        for node in expected_nodes:
            assert result.has_node(node)
        for edge in expected_edges:
            assert result.has_edge(edge[0], edge[1])


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

    def test_index_nodes_empty_types(self, overall_graph):
        with pytest.raises(
            ValueError,
            match="No node types to index",
        ):
            index_nodes([], overall_graph)

    @patch("python.risk_networks.graph_functions.Embedder")
    def test_index_nodes_small_samples(self, mock_embedder, overall_graph_small):
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.embed_store_many.return_value = [[0.1, 0.3], [0.3, 0.4]]

        indexed_node_types = ["TypeA", "TypeB"]
        # Expect ValueError
        with pytest.raises(
            ValueError,
            match="Expected n_neighbors <= n_samples_fit, but n_neighbors = 20, n_samples_fit = 2, n_samples = 2",
        ):
            index_nodes(indexed_node_types, overall_graph_small)

    @patch("python.risk_networks.graph_functions.Embedder")
    def test_index_nodes(self, mock_embedder, overall_graph):
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        # Simulate embedding for the expected number of nodes
        mock_embedder_instance.embed_store_many.return_value = [
            [0.1, 0.3],
        ] * 23

        indexed_node_types = ["TypeA", "TypeB", "TypeC"]
        embedded_texts, nearest_text_distances, nearest_text_indices = index_nodes(
            indexed_node_types, overall_graph
        )

        # Check if the embedded texts are correct
        expected_texts = [
            f"Entity{i}"
            for i in range(1, 31)
            if f"Type{chr(65 + (i % 4))}" in indexed_node_types
        ]
        assert embedded_texts == expected_texts

        # Check the shape of the distances and indices
        expected_shape = (len(expected_texts), 20)
        assert nearest_text_distances.shape == expected_shape
        assert nearest_text_indices.shape == expected_shape
        print("nearest_text_indices", nearest_text_indices)
        print("nearest_text_distances", nearest_text_distances)


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
        progress_callback.assert_called_with(2, 3)

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
        progress_callback.assert_called_with(2, 3)
        progress_callback2.assert_called_with(2, 3)


class TestCreateLinks:
    def test_create_links(self):
        inferred_links = defaultdict(set)
        inferred_links["Entity==PLUS ONE"].add("Entity==PLUS_ONE")
        inferred_links["Entity==PLUS ONE"].add("Entity==ABCDE")
        inferred_links["Entity==PLUS_ONE"].add("Entity==PLUS ONE")
        inferred_links["Entity==ABCDE"].add("Entity==PLUS ONE")

        created_links = create_links(inferred_links)

        expected_links = [
            ("Entity==PLUS ONE", "Entity==PLUS_ONE"),
            ("Entity==ABCDE", "Entity==PLUS ONE"),
        ]
        assert created_links == expected_links

    def test_create_links_return_empty(self):
        inferred_links = defaultdict(set)

        created_links = create_links(inferred_links)

        assert created_links == []
