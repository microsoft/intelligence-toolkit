# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pytest

from intelligence_toolkit.compare_case_groups.prompts import (
    list_prompts,
    report_prompt,
    user_prompt,
)


class TestReportPrompt:
    def test_report_prompt_structure(self) -> None:
        """Test that report_prompt contains expected structure."""
        assert "group comparison report" in report_prompt
        assert "data analyst" in report_prompt
        assert "{description}" in report_prompt
        assert "{filters}" in report_prompt
        assert "{dataset}" in report_prompt

    def test_report_prompt_has_task_section(self) -> None:
        """Test that report_prompt has TASK section."""
        assert "=== TASK ===" in report_prompt
        assert "Dataset description:" in report_prompt
        assert "Group filters:" in report_prompt

    def test_report_prompt_format(self) -> None:
        """Test that report_prompt can be formatted with data."""
        test_description = "Test dataset description"
        test_filters = "Test filters"
        test_dataset = "Test dataset content"
        
        formatted = report_prompt.format(
            description=test_description,
            filters=test_filters,
            dataset=test_dataset,
        )
        
        assert test_description in formatted
        assert test_filters in formatted
        assert test_dataset in formatted
        assert "{description}" not in formatted
        assert "{filters}" not in formatted
        assert "{dataset}" not in formatted

    def test_report_prompt_mentions_requirements(self) -> None:
        """Test that report_prompt mentions key requirements."""
        assert "numeric counts" in report_prompt or "ranks" in report_prompt
        assert "deltas" in report_prompt or "delta" in report_prompt
        assert "examples" in report_prompt or "supported by" in report_prompt


class TestUserPrompt:
    def test_user_prompt_structure(self) -> None:
        """Test that user_prompt contains expected guidance."""
        assert "markdown" in user_prompt
        assert "plain English" in user_prompt or "Plain English" in user_prompt

    def test_user_prompt_mentions_accessibility(self) -> None:
        """Test that user_prompt mentions accessibility requirements."""
        assert (
            "non-native speakers" in user_prompt
            or "non-technical" in user_prompt
        )

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

    def test_all_prompts_are_strings(self) -> None:
        """Test that all prompts in list_prompts are strings."""
        for key, value in list_prompts.items():
            assert isinstance(value, str), f"{key} should be a string"
            assert len(value) > 0, f"{key} should not be empty"
