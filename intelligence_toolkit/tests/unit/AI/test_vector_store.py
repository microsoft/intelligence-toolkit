# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from intelligence_toolkit.AI.vector_store import VectorStore

schema = pa.schema(
    [
        pa.field("hash", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float64())),
        pa.field("additional_details", pa.string()),
    ]
)


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_db_path):
    return VectorStore("test_table", temp_db_path, schema)


def test_vector_store_initialization_without_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir)
        assert store.table is None
        assert store.duckdb_data is None


def test_vector_store_initialization_with_table(temp_db_path):
    store = VectorStore("test_table", temp_db_path, schema)
    assert store.table is not None
    assert store.duckdb_data is not None


def test_vector_store_save(vector_store):
    items = [
        {
            "hash": "hash1",
            "text": "test text 1",
            "vector": [0.1, 0.2, 0.3],
            "additional_details": "{}",
        },
        {
            "hash": "hash2",
            "text": "test text 2",
            "vector": [0.4, 0.5, 0.6],
            "additional_details": "{}",
        },
    ]
    
    vector_store.save(items)
    # If no exception, the save was successful
    assert True


def test_vector_store_save_without_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir)
        
        with pytest.raises(ValueError, match="Table not initialized"):
            store.save([{"hash": "test"}])


def test_vector_store_search_by_column(vector_store):
    items = [
        {
            "hash": "hash1",
            "text": "test text 1",
            "vector": [0.1, 0.2, 0.3],
            "additional_details": "{}",
        },
        {
            "hash": "hash2",
            "text": "test text 2",
            "vector": [0.4, 0.5, 0.6],
            "additional_details": "{}",
        },
    ]
    
    vector_store.save(items)
    vector_store.update_duckdb_data()
    
    result = vector_store.search_by_column("hash1", "hash")
    assert len(result) > 0
    assert result.iloc[0]["hash"] == "hash1"


def test_vector_store_search_by_column_multiple(vector_store):
    items = [
        {
            "hash": "hash1",
            "text": "test text 1",
            "vector": [0.1, 0.2, 0.3],
            "additional_details": "{}",
        },
        {
            "hash": "hash2",
            "text": "test text 2",
            "vector": [0.4, 0.5, 0.6],
            "additional_details": "{}",
        },
    ]
    
    vector_store.save(items)
    vector_store.update_duckdb_data()
    
    result = vector_store.search_by_column(["hash1", "hash2"], "hash")
    assert len(result) >= 2


def test_vector_store_search_by_column_without_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir)
        
        with pytest.raises(ValueError, match="Table not initialized"):
            store.search_by_column("test", "hash")


def test_vector_store_search_by_vector(temp_db_path):
    # LanceDB requires fixed_size_list for vector columns, not variable list
    vector_schema = pa.schema(
        [
            pa.field("hash", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float64(), 3)),  # Fixed size for vector search
            pa.field("additional_details", pa.string()),
        ]
    )
    
    vector_store = VectorStore("test_vector_table", temp_db_path, vector_schema)
    
    items = [
        {
            "hash": "hash1",
            "text": "test text 1",
            "vector": [0.1, 0.2, 0.3],
            "additional_details": "{}",
        },
    ]
    
    vector_store.save(items)
    
    result = vector_store.search_by_vector([0.1, 0.2, 0.3], k=1)
    assert len(result) > 0
    assert "hash" in result[0]


def test_vector_store_search_by_vector_without_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir)
        
        with pytest.raises(ValueError, match="Table not initialized"):
            store.search_by_vector([0.1, 0.2, 0.3])


def test_vector_store_update_duckdb_data(vector_store):
    items = [
        {
            "hash": "hash1",
            "text": "test text 1",
            "vector": [0.1, 0.2, 0.3],
            "additional_details": "{}",
        },
    ]
    
    vector_store.save(items)
    vector_store.update_duckdb_data()
    # If no exception, the update was successful
    assert vector_store.duckdb_data is not None


def test_vector_store_update_duckdb_data_without_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir)
        
        with pytest.raises(ValueError, match="Table not initialized"):
            store.update_duckdb_data()


def test_vector_store_drop_table_without_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir)
        
        with pytest.raises(ValueError, match="Table not initialized"):
            store.drop_table()
