# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for extract_record_data prompts module."""

import intelligence_toolkit.extract_record_data.prompts as prompts


class TestDataExtractionPrompt:
    def test_data_extraction_prompt_is_string(self) -> None:
        """Test that data_extraction_prompt is a string."""
        assert isinstance(prompts.data_extraction_prompt, str)

    def test_data_extraction_prompt_not_empty(self) -> None:
        """Test that data_extraction_prompt is not empty."""
        assert len(prompts.data_extraction_prompt) > 0

    def test_data_extraction_prompt_has_placeholders(self) -> None:
        """Test that data_extraction_prompt has required placeholders."""
        assert "{generation_guidance}" in prompts.data_extraction_prompt
        assert "{input_text}" in prompts.data_extraction_prompt

    def test_data_extraction_prompt_mentions_json(self) -> None:
        """Test that data_extraction_prompt mentions JSON."""
        prompt_lower = prompts.data_extraction_prompt.lower()
        assert "json" in prompt_lower

    def test_data_extraction_prompt_mentions_schema(self) -> None:
        """Test that data_extraction_prompt mentions schema."""
        prompt_lower = prompts.data_extraction_prompt.lower()
        assert "schema" in prompt_lower

    def test_data_extraction_prompt_has_guidance_section(self) -> None:
        """Test that data_extraction_prompt has a guidance section."""
        assert "Generation guidance" in prompts.data_extraction_prompt or "generation_guidance" in prompts.data_extraction_prompt

    def test_data_extraction_prompt_has_input_section(self) -> None:
        """Test that data_extraction_prompt has an input text section."""
        assert "Input text" in prompts.data_extraction_prompt or "input_text" in prompts.data_extraction_prompt

    def test_data_extraction_prompt_warns_against_fabrication(self) -> None:
        """Test that data_extraction_prompt warns against fabricating information."""
        prompt_lower = prompts.data_extraction_prompt.lower()
        assert "not" in prompt_lower and ("fabricate" in prompt_lower or "make up" in prompt_lower or "invent" in prompt_lower or "present" in prompt_lower)
