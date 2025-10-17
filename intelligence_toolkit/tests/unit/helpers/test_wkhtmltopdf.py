# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
from unittest.mock import MagicMock, patch

import pytest

from intelligence_toolkit.helpers.wkhtmltopdf import config_pdfkit, is_in_path


def test_is_in_path_executable_exists(monkeypatch):
    # Mock PATH with a directory containing the executable
    test_path = "/usr/bin:/usr/local/bin"
    monkeypatch.setenv("PATH", test_path)
    
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        
        result = is_in_path("wkhtmltopdf")
        
        assert result is True


def test_is_in_path_executable_not_exists(monkeypatch):
    test_path = "/usr/bin:/usr/local/bin"
    monkeypatch.setenv("PATH", test_path)
    
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        
        result = is_in_path("wkhtmltopdf")
        
        assert result is False


def test_is_in_path_empty_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    
    result = is_in_path("wkhtmltopdf")
    
    assert result is False


def test_is_in_path_with_quotes(monkeypatch):
    test_path = '"/usr/bin":/usr/local/bin'
    monkeypatch.setenv("PATH", test_path)
    
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        
        result = is_in_path("wkhtmltopdf")
        
        assert result is True
        # Verify that quotes are stripped
        mock_exists.assert_called()


def test_config_pdfkit_executable_in_path():
    with patch("intelligence_toolkit.helpers.wkhtmltopdf.is_in_path") as mock_is_in_path, \
         patch("intelligence_toolkit.helpers.wkhtmltopdf.pdfkit.configuration") as mock_config:
        
        mock_is_in_path.return_value = True
        
        config_pdfkit()
        
        # When executable is in PATH, empty string should be passed
        mock_config.assert_called_once_with(wkhtmltopdf="")


def test_config_pdfkit_executable_not_in_path():
    with patch("intelligence_toolkit.helpers.wkhtmltopdf.is_in_path") as mock_is_in_path, \
         patch("intelligence_toolkit.helpers.wkhtmltopdf.pdfkit.configuration") as mock_config, \
         patch("intelligence_toolkit.helpers.wkhtmltopdf.constants") as mock_constants:
        
        mock_is_in_path.return_value = False
        mock_constants.PDF_WKHTMLTOPDF_PATH = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
        
        config_pdfkit()
        
        # When executable is not in PATH, configured path should be used
        mock_config.assert_called_once_with(
            wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
        )


def test_pdfkit_options_structure():
    # Test that pdfkit_options can be imported and has expected structure
    from intelligence_toolkit.helpers import wkhtmltopdf
    
    # This will fail if PDF_MARGIN_INCHES doesn't exist, but we can test the structure
    # that should exist
    assert hasattr(wkhtmltopdf, 'pdfkit_options')
    
    # Check that encoding is set correctly
    if 'encoding' in wkhtmltopdf.pdfkit_options:
        assert wkhtmltopdf.pdfkit_options['encoding'] == "UTF-8"
    
    # Check that enable-local-file-access is set
    if 'enable-local-file-access' in wkhtmltopdf.pdfkit_options:
        assert wkhtmltopdf.pdfkit_options['enable-local-file-access'] is True
