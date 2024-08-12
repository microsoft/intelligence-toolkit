# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import os
import pickle

import pytest

from python.AI.cache_pickle import CachePickle


@pytest.fixture()
def cache_pickle(tmpdir):
    file_name = "test_cache.pickle"
    cache_dir = tmpdir.mkdir("cache")
    cache = CachePickle(file_name, cache_dir)
    yield cache
    cache.reset()


def test_get_all(cache_pickle):
    data = {"key1": "value1", "key2": "value2"}
    with open(cache_pickle.file_path, "wb") as f:
        pickle.dump(data, f)
    result = cache_pickle.get_all()
    assert result == data


def test_get_all_empty(cache_pickle):
    result = cache_pickle.get_all()
    assert result == {}


def test_save(cache_pickle):
    data = {"key1": "value1", "key2": "value2"}
    cache_pickle.save(data)
    with open(cache_pickle.file_path, "rb") as f:
        result = pickle.load(f)
    assert result == data


def test_save_max_size(cache_pickle):
    data = {"key1": "value1", "key2": "value2", "key3": "value3"}
    max_size = 2
    cache_pickle.save(data, max_size)
    with open(cache_pickle.file_path, "rb") as f:
        result = pickle.load(f)
    expected_result = {"key2": "value2", "key3": "value3"}
    assert result == expected_result


def test_reset(cache_pickle):
    data = {"key1": "value1", "key2": "value2"}
    with open(cache_pickle.file_path, "wb") as f:
        pickle.dump(data, f)
    cache_pickle.reset()
    assert not os.path.exists(cache_pickle.file_path)


def test_get(cache_pickle):
    data = {"key1": "value1", "key2": "value2"}
    with open(cache_pickle.file_path, "wb") as f:
        pickle.dump(data, f)
    result = cache_pickle.get("key1", data)
    assert result == "value1"


def test_get_not_found(cache_pickle):
    data = {"key1": "value1", "key2": "value2"}
    with open(cache_pickle.file_path, "wb") as f:
        pickle.dump(data, f)
    result = cache_pickle.get("key3", data)
    assert result == {}


def test_get_non_existing_key(cache_pickle):
    cache_pickle.save({"key1": "value1", "key2": "value2"})
    result = cache_pickle.get("key3", cache_pickle.get_all())
    assert result == {}
