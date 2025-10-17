# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import pytest
from scipy import sparse

from intelligence_toolkit.graph.graph_encoder_embed import GraphEncoderEmbed


@pytest.fixture
def graph_encoder():
    return GraphEncoderEmbed()


@pytest.fixture
def simple_adjacency_matrix():
    """Create a simple 4-node graph adjacency matrix."""
    return sparse.csr_matrix(np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=np.float32))


@pytest.fixture
def simple_labels():
    """Create simple labels for 4 nodes with 2 classes."""
    return np.array([[0], [0], [1], [1]])


def test_graph_encoder_embed_initialization():
    encoder = GraphEncoderEmbed()
    assert encoder is not None


def test_basic_embedding(graph_encoder, simple_adjacency_matrix, simple_labels):
    n = 4
    Z, W = graph_encoder.Basic(simple_adjacency_matrix, simple_labels, n)
    
    # Check output dimensions
    assert Z.shape == (4, 2)  # 4 nodes, 2 classes
    assert W.shape == (4, 2)
    
    # Check that W is sparse
    assert sparse.issparse(W)


def test_basic_embedding_values(graph_encoder, simple_adjacency_matrix, simple_labels):
    n = 4
    Z, W = graph_encoder.Basic(simple_adjacency_matrix, simple_labels, n)
    
    # W should have 1/nk where nk is the count of each class
    # Class 0 has 2 nodes, class 1 has 2 nodes
    W_dense = W.toarray()
    assert W_dense[0, 0] == 0.5  # Node 0 in class 0
    assert W_dense[1, 0] == 0.5  # Node 1 in class 0
    assert W_dense[2, 1] == 0.5  # Node 2 in class 1
    assert W_dense[3, 1] == 0.5  # Node 3 in class 1


def test_basic_with_unlabeled_nodes(graph_encoder, simple_adjacency_matrix):
    n = 4
    labels_with_unlabeled = np.array([[0], [0], [1], [-1]])  # Node 3 is unlabeled
    
    Z, W = graph_encoder.Basic(simple_adjacency_matrix, labels_with_unlabeled, n)
    
    # Check that unlabeled node has zero weights
    W_dense = W.toarray()
    assert np.all(W_dense[3, :] == 0)


def test_diagonal(graph_encoder, simple_adjacency_matrix):
    n = 4
    X_diag = graph_encoder.Diagonal(simple_adjacency_matrix, n)
    
    # Check that diagonal is all 1s
    X_dense = X_diag.toarray()
    assert np.all(np.diag(X_dense) == 1)


def test_laplacian(graph_encoder, simple_adjacency_matrix):
    n = 4
    L = graph_encoder.Laplacian(simple_adjacency_matrix, n)
    
    # Check output shape
    assert L.shape == (4, 4)
    
    # Check that it's sparse
    assert sparse.issparse(L)
    
    # Laplacian normalization should preserve symmetry
    L_dense = L.toarray()
    assert np.allclose(L_dense, L_dense.T)


def test_correlation(graph_encoder):
    # Create a simple embedding matrix
    Z = sparse.csr_matrix(np.array([
        [3, 4],
        [1, 0],
        [0, 1]
    ], dtype=np.float32))
    
    Z_norm = graph_encoder.Correlation(Z)
    
    # Check that rows are normalized (each row should have norm 1)
    Z_norm_dense = Z_norm.toarray()
    row_norms = np.linalg.norm(Z_norm_dense, axis=1)
    assert np.allclose(row_norms, 1.0)


def test_correlation_with_zero_rows(graph_encoder):
    # Test with a zero row
    Z = sparse.csr_matrix(np.array([
        [3, 4],
        [0, 0],  # Zero row
        [1, 1]
    ], dtype=np.float32))
    
    Z_norm = graph_encoder.Correlation(Z)
    
    # Zero rows should remain zero (nan_to_num handles division by zero)
    Z_norm_dense = Z_norm.toarray()
    assert np.all(Z_norm_dense[1, :] == 0)


def test_edge_list_size_s2(graph_encoder):
    # S2 edge list (2 columns)
    edge_list = np.array([[0, 1], [1, 2], [2, 3]])
    
    result = graph_encoder.edge_list_size(edge_list)
    assert result == "S2"


def test_edge_list_size_s3(graph_encoder):
    # S3 edge list (3 columns)
    edge_list = np.array([[0, 1, 0.5], [1, 2, 0.8], [2, 3, 1.0]])
    
    result = graph_encoder.edge_list_size(edge_list)
    assert result == "S3"


def test_edge_to_sparse_s2(graph_encoder):
    edge_list = np.array([[0, 1], [1, 2], [2, 0]])
    n = 3
    
    X_sparse = graph_encoder.Edge_to_Sparse(edge_list, n, "S2")
    
    # Check shape and type
    assert X_sparse.shape == (3, 3)
    assert sparse.issparse(X_sparse)
    
    # Check values (all should be 1 for S2)
    X_dense = X_sparse.toarray()
    assert X_dense[0, 1] == 1
    assert X_dense[1, 2] == 1
    assert X_dense[2, 0] == 1


def test_edge_to_sparse_s3(graph_encoder):
    edge_list = np.array([[0, 1, 0.5], [1, 2, 0.8], [2, 0, 1.2]])
    n = 3
    
    X_sparse = graph_encoder.Edge_to_Sparse(edge_list, n, "S3")
    
    # Check that weights are preserved
    X_dense = X_sparse.toarray()
    assert np.isclose(X_dense[0, 1], 0.5)
    assert np.isclose(X_dense[1, 2], 0.8)
    assert np.isclose(X_dense[2, 0], 1.2)


def test_run_with_edge_list(graph_encoder):
    # Use S3 format (with weights) so edge_list_size detection works correctly
    edge_list = np.array([[0, 1, 1.0], [1, 2, 1.0], [2, 0, 1.0]])
    labels = np.array([[0], [1], [0]])
    n = 3
    
    Z, W = graph_encoder.run(edge_list, labels, n, EdgeList=True)
    
    # Check output dimensions
    assert Z.shape == (3, 2)  # 3 nodes, 2 classes
    assert W.shape == (3, 2)


def test_run_with_all_options(graph_encoder):
    edge_list = np.array([[0, 1, 1.0], [1, 2, 1.0], [2, 0, 1.0]])
    labels = np.array([[0], [1], [0]])
    n = 3
    
    Z, W = graph_encoder.run(
        edge_list,
        labels,
        n,
        EdgeList=True,
        DiagA=True,
        Laplacian=True,
        Correlation=True
    )
    
    # Should complete without errors
    assert Z.shape == (3, 2)


def test_run_without_correlation(graph_encoder, simple_adjacency_matrix, simple_labels):
    n = 4
    
    Z, W = graph_encoder.run(
        simple_adjacency_matrix,
        simple_labels,
        n,
        EdgeList=False,
        Correlation=False
    )
    
    # Without correlation, rows won't be normalized
    assert Z.shape == (4, 2)


def test_basic_with_multiple_classes(graph_encoder, simple_adjacency_matrix):
    n = 4
    labels = np.array([[0], [1], [2], [0]])  # 3 classes
    
    Z, W = graph_encoder.Basic(simple_adjacency_matrix, labels, n)
    
    assert Z.shape == (4, 3)  # 4 nodes, 3 classes
    assert W.shape == (4, 3)


def test_diagonal_preserves_sparsity(graph_encoder):
    # Very sparse matrix
    X = sparse.csr_matrix(np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32))
    
    X_diag = graph_encoder.Diagonal(X, 4)
    
    # Should still be sparse
    assert sparse.issparse(X_diag)
    
    # Should have more non-zero elements (added diagonal)
    assert X_diag.nnz >= X.nnz
