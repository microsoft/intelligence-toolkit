# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for extract_record_data __init__ module."""

from unittest.mock import mock_open, patch

import intelligence_toolkit.extract_record_data as extract_record_data


class TestGetReadme:
    def test_get_readme_returns_string(self) -> None:
        """Test that get_readme returns a string."""
        mock_data = "# Extract Record Data\n\nThis is a test readme."
        with patch("builtins.open", mock_open(read_data=mock_data)):
            result = extract_record_data.get_readme()
            assert isinstance(result, str)
            assert len(result) > 0

    def test_get_readme_opens_correct_file(self) -> None:
        """Test that get_readme opens the README.md file in the correct location."""
        mock_data = "Test content"
        m = mock_open(read_data=mock_data)
        with patch("builtins.open", m):
            extract_record_data.get_readme()
            m.assert_called_once()
            # Check that the path ends with README.md
            called_path = m.call_args[0][0]
            assert called_path.endswith("README.md")

    def test_get_readme_content(self) -> None:
        """Test that get_readme returns the file content."""
        mock_data = "# Extract Record Data\n\nExtract structured data from text."
        with patch("builtins.open", mock_open(read_data=mock_data)):
            result = extract_record_data.get_readme()
            assert result == mock_data
