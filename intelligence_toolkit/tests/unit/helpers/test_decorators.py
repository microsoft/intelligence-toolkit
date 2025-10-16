# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import time
from unittest.mock import MagicMock

import pytest

from intelligence_toolkit.helpers.decorators import retry_with_backoff


def test_retry_with_backoff_success_on_first_try():
    mock_func = MagicMock(return_value="success")
    decorated = retry_with_backoff()(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_with_backoff_success_after_retries():
    mock_func = MagicMock(side_effect=[Exception("Error 1"), Exception("Error 2"), "success"])
    decorated = retry_with_backoff(retries=5, backoff_in_seconds=0.01)(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_with_backoff_max_retries_exceeded():
    mock_func = MagicMock(side_effect=Exception("Persistent error"))
    decorated = retry_with_backoff(retries=3, backoff_in_seconds=0.01)(mock_func)
    
    with pytest.raises(Exception, match="Persistent error"):
        decorated()
    
    assert mock_func.call_count == 4  # Initial call + 3 retries


def test_retry_with_backoff_preserves_function_name():
    def test_function():
        return "test"
    
    decorated = retry_with_backoff()(test_function)
    
    assert decorated.__name__ == "test_function"


def test_retry_with_backoff_with_arguments():
    mock_func = MagicMock(side_effect=[Exception("Error"), "success"])
    decorated = retry_with_backoff(retries=3, backoff_in_seconds=0.01)(mock_func)
    
    result = decorated("arg1", "arg2", kwarg1="value1")
    
    assert result == "success"
    assert mock_func.call_count == 2
    mock_func.assert_called_with("arg1", "arg2", kwarg1="value1")


def test_retry_with_backoff_exponential_backoff():
    mock_func = MagicMock(side_effect=[Exception("Error 1"), Exception("Error 2"), "success"])
    
    start_time = time.time()
    decorated = retry_with_backoff(retries=5, backoff_in_seconds=0.1)(mock_func)
    result = decorated()
    elapsed = time.time() - start_time
    
    assert result == "success"
    # Should have some delay due to backoff (0.1 * 2^0 + 0.1 * 2^1 ≈ 0.3 seconds minimum)
    assert elapsed >= 0.1


def test_retry_with_backoff_no_retries():
    mock_func = MagicMock(side_effect=Exception("Immediate failure"))
    decorated = retry_with_backoff(retries=0, backoff_in_seconds=0.01)(mock_func)
    
    with pytest.raises(Exception, match="Immediate failure"):
        decorated()
    
    assert mock_func.call_count == 1


def test_retry_with_backoff_returns_correct_value():
    expected_value = {"key": "value", "number": 42}
    mock_func = MagicMock(return_value=expected_value)
    decorated = retry_with_backoff()(mock_func)
    
    result = decorated()
    
    assert result == expected_value
