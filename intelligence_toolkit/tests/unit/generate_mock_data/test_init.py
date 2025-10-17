# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import os
from unittest.mock import patch, mock_open
from intelligence_toolkit.generate_mock_data import get_readme


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "../../../generate_mock_data/README.md",
        )
    ),
    reason="README.md not found",
)
def test_get_readme_returns_string():
    result = get_readme()
    assert isinstance(result, str)


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "../../../generate_mock_data/README.md",
        )
    ),
    reason="README.md not found",
)
def test_get_readme_reads_file():
    result = get_readme()
    # Should return content from README.md
    assert len(result) > 0


@patch("builtins.open", new_callable=mock_open, read_data="# Test README\nThis is test content")
def test_get_readme_opens_correct_file(mock_file):
    result = get_readme()
    
    assert result == "# Test README\nThis is test content"
    # Verify it tried to open a README.md file
    mock_file.assert_called_once()
    call_args = mock_file.call_args[0][0]
    assert "README.md" in call_args


@patch("builtins.open", new_callable=mock_open, read_data="Mock content")
def test_get_readme_uses_correct_path(mock_file):
    result = get_readme()
    
    # Should use os.path.join with __file__ directory
    call_args = mock_file.call_args[0][0]
    assert os.path.isabs(call_args) or "README.md" in call_args
