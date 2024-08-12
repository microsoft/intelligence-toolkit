# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import pytest

from python.AI.utils import (
    get_token_count,
    hash_text,
    prepare_messages,
    prepare_validation,
    try_parse_json_object,
)


def test_get_token_count():
    text = "example text"
    result = get_token_count(text)
    expected = 4

    assert result == expected


def test_get_token_count_with_model():
    text = "example text"
    result = get_token_count(text, None, "gpt-4o")
    expected = 4
    assert result == expected


def test_get_token_count_with_encoding():
    text = "example text"
    result = get_token_count(text, "o200k_base")
    expected = 4
    assert result == expected


def test_hash_text():
    text = "example test \n\n example test two"
    hash_returned = hash_text(text)
    assert (
        hash_returned
        == "5aaa5f424725140ddfeb0f5747e7543a52dc18ebde854b83e554f155ba2abfab"
    )


def test_try_parse_json_object_ok():
    obj_test = '{"key": "value"}'
    result = try_parse_json_object(obj_test)
    assert isinstance(result, dict)


def test_try_parse_json_object_exception():
    obj_test = {"key": "value"}
    with pytest.raises(Exception):
        try_parse_json_object(obj_test)


def test_prepare_validation():
    messages = [{"role": "user", "content": "Write me a poem about a cat"}]
    report = "The cat poem: The cat is a furry animal that likes to play with yarn. It is a pet that is loved by many people. The cat is a furry animal that likes to play with yarn. It is a pet that is loved by many people."
    message = prepare_validation(messages, report)
    expected = "Write me a poem about a cat"
    assert expected in message[0]["content"]


def test_prepare_messages():
    variables = {"animal": "cat", "place": "tree"}
    system_message = "I can write you a poem about a {animal} in a {place}"
    message = prepare_messages(system_message, variables)
    expected = "I can write you a poem about a cat in a tree"
    assert message[0]["content"] == expected


def test_prepare_messages_user():
    variables = {"animal": "cat", "place": "tree", "rescue": "fireman"}
    system_message = "I can write you a poem about a {animal} in a {place}"
    user_message = "Make the {rescue} save the cat"
    message = prepare_messages(system_message, variables, user_message)
    assert message[0]["content"] == "I can write you a poem about a cat in a tree"
    assert message[1]["content"] == "Make the fireman save the cat"
