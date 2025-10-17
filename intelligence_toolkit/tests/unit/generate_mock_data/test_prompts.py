# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
from intelligence_toolkit.generate_mock_data.prompts import (
    unseeded_data_generation_prompt,
    seeded_data_generation_prompt,
    text_generation_prompt,
)


def test_unseeded_data_generation_prompt_exists():
    assert isinstance(unseeded_data_generation_prompt, str)
    assert len(unseeded_data_generation_prompt) > 0


def test_unseeded_data_generation_prompt_has_placeholders():
    assert "{generation_guidance}" in unseeded_data_generation_prompt
    assert "{primary_record_array}" in unseeded_data_generation_prompt
    assert "{total_records}" in unseeded_data_generation_prompt


def test_unseeded_data_generation_prompt_has_instructions():
    assert "JSON object" in unseeded_data_generation_prompt
    assert "schema" in unseeded_data_generation_prompt.lower()


def test_seeded_data_generation_prompt_exists():
    assert isinstance(seeded_data_generation_prompt, str)
    assert len(seeded_data_generation_prompt) > 0


def test_seeded_data_generation_prompt_has_placeholders():
    assert "{generation_guidance}" in seeded_data_generation_prompt
    assert "{primary_record_array}" in seeded_data_generation_prompt
    assert "{seed_record}" in seeded_data_generation_prompt
    assert "{record_targets}" in seeded_data_generation_prompt


def test_seeded_data_generation_prompt_mentions_duplicates():
    assert "duplicate" in seeded_data_generation_prompt.lower()
    assert "relation" in seeded_data_generation_prompt.lower()


def test_text_generation_prompt_exists():
    assert isinstance(text_generation_prompt, str)
    assert len(text_generation_prompt) > 0


def test_text_generation_prompt_has_placeholders():
    assert "{generation_guidance}" in text_generation_prompt
    assert "{input_text}" in text_generation_prompt


def test_text_generation_prompt_mentions_document():
    assert "text" in text_generation_prompt.lower() or "document" in text_generation_prompt.lower()


def test_prompts_are_distinct():
    # Verify each prompt is unique
    assert unseeded_data_generation_prompt != seeded_data_generation_prompt
    assert unseeded_data_generation_prompt != text_generation_prompt
    assert seeded_data_generation_prompt != text_generation_prompt
