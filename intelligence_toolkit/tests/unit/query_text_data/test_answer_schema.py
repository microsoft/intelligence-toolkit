# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data answer_schema module."""

from intelligence_toolkit.query_text_data.answer_schema import (
    theme_integration_format,
    theme_summarization_format,
    thematic_update_format,
)


class TestThemeIntegrationFormat:
    def test_structure(self) -> None:
        """Test theme_integration_format has correct structure."""
        assert "type" in theme_integration_format
        assert "json_schema" in theme_integration_format
        assert theme_integration_format["type"] == "json_schema"

    def test_schema_properties(self) -> None:
        """Test theme_integration_format schema properties."""
        schema = theme_integration_format["json_schema"]["schema"]
        assert "properties" in schema
        properties = schema["properties"]
        
        assert "report_title" in properties
        assert "report_overview" in properties
        assert "report_implications" in properties
        assert "answer" in properties

    def test_required_fields(self) -> None:
        """Test theme_integration_format required fields."""
        schema = theme_integration_format["json_schema"]["schema"]
        required = schema["required"]
        
        assert "report_title" in required
        assert "report_overview" in required
        assert "report_implications" in required
        assert "answer" in required


class TestThemeSummarizationFormat:
    def test_structure(self) -> None:
        """Test theme_summarization_format has correct structure."""
        assert "type" in theme_summarization_format
        assert "json_schema" in theme_summarization_format
        assert theme_summarization_format["type"] == "json_schema"

    def test_schema_properties(self) -> None:
        """Test theme_summarization_format schema properties."""
        schema = theme_summarization_format["json_schema"]["schema"]
        assert "properties" in schema
        properties = schema["properties"]
        
        assert "theme_title" in properties
        assert "theme_points" in properties

    def test_theme_points_structure(self) -> None:
        """Test theme_points array structure."""
        schema = theme_summarization_format["json_schema"]["schema"]
        theme_points = schema["properties"]["theme_points"]
        
        assert theme_points["type"] == "array"
        assert "items" in theme_points
        
        item_properties = theme_points["items"]["properties"]
        assert "point_title" in item_properties
        assert "point_evidence" in item_properties
        assert "point_commentary" in item_properties

    def test_required_fields(self) -> None:
        """Test theme_summarization_format required fields."""
        schema = theme_summarization_format["json_schema"]["schema"]
        required = schema["required"]
        
        assert "theme_title" in required
        assert "theme_points" in required


class TestThematicUpdateFormat:
    def test_structure(self) -> None:
        """Test thematic_update_format has correct structure."""
        assert "type" in thematic_update_format
        assert "json_schema" in thematic_update_format
        assert thematic_update_format["type"] == "json_schema"

    def test_schema_properties(self) -> None:
        """Test thematic_update_format schema properties."""
        schema = thematic_update_format["json_schema"]["schema"]
        assert "properties" in schema
        properties = schema["properties"]
        
        assert "updates" in properties
        assert "themes" in properties

    def test_updates_structure(self) -> None:
        """Test updates array structure."""
        schema = thematic_update_format["json_schema"]["schema"]
        updates = schema["properties"]["updates"]
        
        assert updates["type"] == "array"
        assert "items" in updates
        
        item_properties = updates["items"]["properties"]
        assert "point_id" in item_properties
        assert "point_title" in item_properties
        assert "source_ids" in item_properties

    def test_themes_structure(self) -> None:
        """Test themes array structure."""
        schema = thematic_update_format["json_schema"]["schema"]
        themes = schema["properties"]["themes"]
        
        assert themes["type"] == "array"
        assert "items" in themes
        
        item_properties = themes["items"]["properties"]
        assert "theme_title" in item_properties
        assert "point_ids" in item_properties

    def test_required_fields(self) -> None:
        """Test thematic_update_format required fields."""
        schema = thematic_update_format["json_schema"]["schema"]
        required = schema["required"]
        
        assert "updates" in required
        assert "themes" in required

    def test_strict_mode(self) -> None:
        """Test that all schemas have strict mode enabled."""
        assert theme_integration_format["json_schema"]["strict"] is True
        assert theme_summarization_format["json_schema"]["strict"] is True
        assert thematic_update_format["json_schema"]["strict"] is True
