# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pytest

from intelligence_toolkit.match_entity_records.prompts import (
    list_prompts,
    report_prompt,
    user_prompt,
)


class TestReportPrompt:
    def test_report_prompt_structure(self) -> None:
        """Test that report_prompt contains expected structure."""
        assert "Goal:" in report_prompt
        assert "RELATEDNESS" in report_prompt
        assert "scale of 0-10" in report_prompt
        assert "{data}" in report_prompt
        assert "Group ID" in report_prompt
        assert "Relatedness" in report_prompt
        assert "Explanation" in report_prompt

    def test_report_prompt_format(self) -> None:
        """Test that report_prompt can be formatted with data."""
        test_data = "test_data_value"
        formatted = report_prompt.format(data=test_data)
        assert "test_data_value" in formatted
        assert "{data}" not in formatted


class TestUserPrompt:
    def test_user_prompt_structure(self) -> None:
        """Test that user_prompt contains expected guidance."""
        assert "Factors indicating unrelatedness" in user_prompt
        assert "Factors indicating relatedness" in user_prompt
        assert "Factors that should be ignored" in user_prompt
        assert "Factors that should be considered" in user_prompt
        assert "spelling" in user_prompt
        assert "formatting" in user_prompt
        assert "missing values" in user_prompt

    def test_user_prompt_is_string(self) -> None:
        """Test that user_prompt is a non-empty string."""
        assert isinstance(user_prompt, str)
        assert len(user_prompt) > 0


class TestListPrompts:
    def test_list_prompts_structure(self) -> None:
        """Test that list_prompts contains expected keys."""
        assert "report_prompt" in list_prompts
        assert "user_prompt" in list_prompts
        assert "safety_prompt" in list_prompts

    def test_list_prompts_values(self) -> None:
        """Test that list_prompts contains correct values."""
        assert list_prompts["report_prompt"] == report_prompt
        assert list_prompts["user_prompt"] == user_prompt
        assert isinstance(list_prompts["safety_prompt"], str)

    def test_safety_prompt_content(self) -> None:
        """Test that safety_prompt contains expected content."""
        safety_prompt = list_prompts["safety_prompt"]
        assert len(safety_prompt) > 0
        assert isinstance(safety_prompt, str)
