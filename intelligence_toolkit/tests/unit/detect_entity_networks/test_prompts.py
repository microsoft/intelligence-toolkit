# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import pytest

from intelligence_toolkit.detect_entity_networks.prompts import (
    list_prompts,
    report_prompt,
    user_prompt,
)


class TestReportPrompt:
    def test_report_prompt_structure(self) -> None:
        """Test that report_prompt contains expected structure."""
        assert "entity network" in report_prompt
        assert "flag exposure" in report_prompt
        assert "{entity_id}" in report_prompt
        assert "{network_id}" in report_prompt
        assert "{network_nodes}" in report_prompt
        assert "{network_edges}" in report_prompt
        assert "{exposure}" in report_prompt

    def test_report_prompt_has_calibration_info(self) -> None:
        """Test that report_prompt includes calibration variables."""
        assert "{max_flags}" in report_prompt
        assert "{mean_flags}" in report_prompt

    def test_report_prompt_has_task_section(self) -> None:
        """Test that report_prompt has TASK section."""
        assert "=== TASK ===" in report_prompt
        assert "Selected entity:" in report_prompt
        assert "Selected network:" in report_prompt

    def test_report_prompt_format(self) -> None:
        """Test that report_prompt can be formatted with data."""
        formatted = report_prompt.format(
            entity_id="E123",
            network_id=42,
            network_nodes="node1,node2",
            network_edges="edge1,edge2",
            exposure="test_exposure",
            max_flags=10,
            mean_flags=5.5,
        )
        
        assert "E123" in formatted
        assert "42" in formatted
        assert "node1,node2" in formatted
        assert "edge1,edge2" in formatted
        assert "test_exposure" in formatted
        assert "10" in formatted
        assert "5.5" in formatted

    def test_report_prompt_mentions_connections(self) -> None:
        """Test that report_prompt discusses connection analysis."""
        assert "connections" in report_prompt or "connected" in report_prompt
        assert "entities" in report_prompt
        assert "attributes" in report_prompt or "shared attributes" in report_prompt

    def test_report_prompt_has_heading_instructions(self) -> None:
        """Test that report_prompt includes heading format instructions."""
        assert "Evaluation of" in report_prompt
        assert "Entity ID" in report_prompt or "<Entity ID>" in report_prompt
        assert "Network" in report_prompt


class TestUserPrompt:
    def test_user_prompt_structure(self) -> None:
        """Test that user_prompt contains expected guidance."""
        assert "Goal" in user_prompt or "goal" in user_prompt
        assert "entity" in user_prompt
        assert "flag exposure" in user_prompt

    def test_user_prompt_mentions_evaluation(self) -> None:
        """Test that user_prompt discusses evaluation tasks."""
        assert "Evaluate" in user_prompt or "evaluate" in user_prompt
        assert "likelihood" in user_prompt or "same" in user_prompt

    def test_user_prompt_mentions_accessibility(self) -> None:
        """Test that user_prompt mentions accessibility requirements."""
        assert "markdown" in user_prompt
        assert (
            "plain English" in user_prompt
            or "non-native speakers" in user_prompt
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
