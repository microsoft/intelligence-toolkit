# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
from unittest.mock import MagicMock

import pytest

from intelligence_toolkit.AI.base_batch_async import BaseBatchAsync
from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback


@pytest.fixture
def base_batch():
    return BaseBatchAsync()


def test_base_batch_async_initialization():
    batch = BaseBatchAsync()
    assert batch.total_tasks == 1
    assert batch.completed_tasks == 0
    assert batch.previous_completed_tasks == 0


def test_base_batch_async_progress_callback(base_batch):
    assert base_batch.completed_tasks == 0
    base_batch.progress_callback()
    assert base_batch.completed_tasks == 1
    base_batch.progress_callback()
    assert base_batch.completed_tasks == 2


@pytest.mark.asyncio
async def test_track_progress_with_tasks():
    batch = BaseBatchAsync()
    batch.total_tasks = 3
    
    callback = MagicMock(spec=ProgressBatchCallback)
    callback.on_batch_change = MagicMock()
    
    # Create some mock tasks
    async def mock_task():
        await asyncio.sleep(0.01)
        batch.progress_callback()
    
    tasks = [asyncio.create_task(mock_task()) for _ in range(3)]
    
    # Track progress
    await batch.track_progress(tasks, [callback])
    
    # Verify callback was called
    assert callback.on_batch_change.called
    assert batch.completed_tasks == 3


@pytest.mark.asyncio
async def test_track_progress_multiple_callbacks():
    batch = BaseBatchAsync()
    batch.total_tasks = 2
    
    callback1 = MagicMock(spec=ProgressBatchCallback)
    callback1.on_batch_change = MagicMock()
    callback2 = MagicMock(spec=ProgressBatchCallback)
    callback2.on_batch_change = MagicMock()
    
    async def mock_task():
        await asyncio.sleep(0.01)
        batch.progress_callback()
    
    tasks = [asyncio.create_task(mock_task()) for _ in range(2)]
    
    await batch.track_progress(tasks, [callback1, callback2])
    
    assert callback1.on_batch_change.called
    assert callback2.on_batch_change.called
    assert batch.completed_tasks == 2


@pytest.mark.asyncio
async def test_track_progress_completed_immediately():
    batch = BaseBatchAsync()
    batch.total_tasks = 1
    batch.completed_tasks = 1
    
    callback = MagicMock(spec=ProgressBatchCallback)
    callback.on_batch_change = MagicMock()
    
    # Create already completed tasks
    async def completed_task():
        pass
    
    task = asyncio.create_task(completed_task())
    await asyncio.sleep(0.01)  # Let task complete
    
    await batch.track_progress([task], [callback])
    
    assert callback.on_batch_change.called


@pytest.mark.asyncio
async def test_track_progress_no_change():
    batch = BaseBatchAsync()
    batch.total_tasks = 0
    batch.completed_tasks = 0
    
    callback = MagicMock(spec=ProgressBatchCallback)
    callback.on_batch_change = MagicMock()
    
    # Empty task list
    await batch.track_progress([], [callback])
    
    # Should still call callback at the end
    assert callback.on_batch_change.called
