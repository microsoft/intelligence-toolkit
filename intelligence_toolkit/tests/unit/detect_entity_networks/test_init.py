# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from intelligence_toolkit.detect_entity_networks import get_readme


class TestGetReadme:
    def test_get_readme_success(self) -> None:
        """Test get_readme successfully reads README.md."""
        mock_readme_content = "# Detect Entity Networks\n\nThis is a test README."

        with patch("builtins.open", mock_open(read_data=mock_readme_content)):
            result = get_readme()

        assert result == mock_readme_content
        assert "Detect Entity Networks" in result

    def test_get_readme_file_structure(self) -> None:
        """Test that get_readme reads from correct path."""
        mock_content = "Test content"

        with patch("builtins.open", mock_open(read_data=mock_content)) as mock_file:
            get_readme()
            # Verify open was called
            mock_file.assert_called_once()
            # Get the path that was used
            call_args = mock_file.call_args[0][0]
            # Should end with README.md
            assert str(call_args).endswith("README.md")

    def test_get_readme_returns_string(self) -> None:
        """Test that get_readme returns a string."""
        mock_content = "Test README content"

        with patch("builtins.open", mock_open(read_data=mock_content)):
            result = get_readme()

        assert isinstance(result, str)
        assert len(result) > 0
