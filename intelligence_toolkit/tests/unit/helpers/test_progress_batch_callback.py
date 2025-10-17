# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pytest

from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback


def test_progress_batch_callback_initialization():
    callback = ProgressBatchCallback()
    
    assert callback.current_batch == 0
    assert callback.total_batches == 0


def test_progress_batch_callback_on_batch_change():
    callback = ProgressBatchCallback()
    
    callback.on_batch_change(5, 10)
    
    assert callback.current_batch == 5
    assert callback.total_batches == 10


def test_progress_batch_callback_on_batch_change_with_message():
    callback = ProgressBatchCallback()
    
    callback.on_batch_change(3, 7, message="Processing...")
    
    assert callback.current_batch == 3
    assert callback.total_batches == 7
    assert callback.message == "Processing..."


def test_progress_batch_callback_multiple_updates():
    callback = ProgressBatchCallback()
    
    callback.on_batch_change(1, 10)
    assert callback.current_batch == 1
    
    callback.on_batch_change(5, 10)
    assert callback.current_batch == 5
    
    callback.on_batch_change(10, 10)
    assert callback.current_batch == 10


def test_progress_batch_callback_message_attribute_exists():
    callback = ProgressBatchCallback()
    callback.on_batch_change(1, 1, message="test")
    
    assert hasattr(callback, 'message')
    assert callback.message == "test"


def test_progress_batch_callback_default_message():
    callback = ProgressBatchCallback()
    callback.on_batch_change(1, 1)
    
    # Default message is empty string
    assert callback.message == ""
