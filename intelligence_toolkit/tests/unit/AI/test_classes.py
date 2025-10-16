# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pytest

from intelligence_toolkit.AI.classes import LLMCallback, VectorData


def test_llm_callback_initialization():
    callback = LLMCallback()
    assert callback.response == []


def test_llm_callback_on_llm_new_token():
    callback = LLMCallback()
    callback.on_llm_new_token("Hello")
    assert callback.response == ["Hello"]
    
    callback.on_llm_new_token(" World")
    assert callback.response == ["Hello", " World"]


def test_llm_callback_multiple_tokens():
    callback = LLMCallback()
    tokens = ["The", " quick", " brown", " fox"]
    for token in tokens:
        callback.on_llm_new_token(token)
    assert callback.response == tokens


def test_vector_data_structure():
    # Test that VectorData is a properly defined class with annotations
    # VectorData uses type annotations but doesn't set default values
    assert hasattr(VectorData, '__annotations__')
    assert 'hash' in VectorData.__annotations__
    assert 'text' in VectorData.__annotations__
    assert 'vector' in VectorData.__annotations__
    assert 'additional_details' in VectorData.__annotations__


def test_vector_data_assignment():
    vector_data = VectorData()
    vector_data.hash = "test_hash_123"
    vector_data.text = "Sample text for testing"
    vector_data.vector = [0.1, 0.2, 0.3, 0.4]
    vector_data.additional_details = {"key": "value"}
    
    assert vector_data.hash == "test_hash_123"
    assert vector_data.text == "Sample text for testing"
    assert vector_data.vector == [0.1, 0.2, 0.3, 0.4]
    assert vector_data.additional_details == {"key": "value"}
