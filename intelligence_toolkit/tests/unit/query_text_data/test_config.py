# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Tests for query_text_data config module."""

import intelligence_toolkit.query_text_data.config as config


class TestCacheName:
    def test_cache_name_is_string(self) -> None:
        """Test that cache_name is a string."""
        assert isinstance(config.cache_name, str)

    def test_cache_name_not_empty(self) -> None:
        """Test that cache_name is not empty."""
        assert len(config.cache_name) > 0

    def test_cache_name_value(self) -> None:
        """Test that cache_name has the expected value."""
        assert config.cache_name == "query_text_data"
