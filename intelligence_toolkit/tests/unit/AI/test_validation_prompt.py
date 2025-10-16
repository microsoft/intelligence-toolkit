# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from intelligence_toolkit.AI.validation_prompt import GROUNDEDNESS_PROMPT


def test_groundedness_prompt_exists():
    assert GROUNDEDNESS_PROMPT is not None
    assert isinstance(GROUNDEDNESS_PROMPT, str)
    assert len(GROUNDEDNESS_PROMPT) > 0


def test_groundedness_prompt_contains_key_instructions():
    assert "AI assistant" in GROUNDEDNESS_PROMPT
    assert "coherence" in GROUNDEDNESS_PROMPT
    assert "report instructions" in GROUNDEDNESS_PROMPT
    assert "generated report" in GROUNDEDNESS_PROMPT


def test_groundedness_prompt_contains_rating_scale():
    assert "5:" in GROUNDEDNESS_PROMPT
    assert "1:" in GROUNDEDNESS_PROMPT
    assert "score" in GROUNDEDNESS_PROMPT.lower()
    assert "explanation" in GROUNDEDNESS_PROMPT.lower()


def test_groundedness_prompt_contains_json_format_instruction():
    assert "JSON format" in GROUNDEDNESS_PROMPT
    assert "score" in GROUNDEDNESS_PROMPT
    assert "explanation" in GROUNDEDNESS_PROMPT


def test_groundedness_prompt_contains_examples():
    assert "Example Task" in GROUNDEDNESS_PROMPT
    assert "REPORT INSTRUCTIONS" in GROUNDEDNESS_PROMPT
    assert "GENERATED REPORT" in GROUNDEDNESS_PROMPT


def test_groundedness_prompt_mentions_data_accuracy():
    assert "accurately" in GROUNDEDNESS_PROMPT.lower() or "accurate" in GROUNDEDNESS_PROMPT.lower()
    assert "data" in GROUNDEDNESS_PROMPT.lower()
