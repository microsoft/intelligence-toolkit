# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from intelligence_toolkit.AI.metaprompts import (
    do_not_disrespect_context,
    do_not_harm,
    do_not_harm_question_answering,
)


def test_do_not_harm_question_answering_exists():
    assert do_not_harm_question_answering is not None
    assert isinstance(do_not_harm_question_answering, str)
    assert len(do_not_harm_question_answering) > 0


def test_do_not_harm_question_answering_contains_key_phrases():
    assert "Decline to answer" in do_not_harm_question_answering
    assert "Never" in do_not_harm_question_answering
    assert "speculate" in do_not_harm_question_answering
    assert "relevant documents" in do_not_harm_question_answering


def test_do_not_harm_exists():
    assert do_not_harm is not None
    assert isinstance(do_not_harm, str)
    assert len(do_not_harm) > 0


def test_do_not_harm_contains_key_phrases():
    assert "must not generate content" in do_not_harm
    assert "harmful" in do_not_harm
    assert "hateful" in do_not_harm
    assert "racist" in do_not_harm
    assert "sexist" in do_not_harm
    assert "violent" in do_not_harm


def test_do_not_disrespect_context_exists():
    assert do_not_disrespect_context is not None
    assert isinstance(do_not_disrespect_context, str)
    assert len(do_not_disrespect_context) > 0


def test_do_not_disrespect_context_contains_key_phrases():
    assert "tone of the document" in do_not_disrespect_context
    assert "vague" in do_not_disrespect_context
    assert "controversial" in do_not_disrespect_context
    assert "dates and times" in do_not_disrespect_context
    assert "professional conversation" in do_not_disrespect_context


def test_all_metaprompts_are_strings():
    prompts = [
        do_not_harm_question_answering,
        do_not_harm,
        do_not_disrespect_context,
    ]
    for prompt in prompts:
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0
