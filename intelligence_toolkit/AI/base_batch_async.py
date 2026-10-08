# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import logging

from intelligence_toolkit.AI.defaults import DEFAULT_CONCURRENT_COROUTINES
from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback

logger = logging.getLogger(__name__)


class BaseBatchAsync:
    total_tasks: int = 1
    completed_tasks: int = 0
    previous_completed_tasks: int = 0
    _concurrency: int = DEFAULT_CONCURRENT_COROUTINES
    _semaphore: asyncio.Semaphore | None = None
    _semaphore_loop: asyncio.AbstractEventLoop | None = None

    def set_concurrency(self, concurrent_coroutines: int | None) -> None:
        if concurrent_coroutines is None:
            concurrent_coroutines = DEFAULT_CONCURRENT_COROUTINES
        self._concurrency = max(1, int(concurrent_coroutines))
        self._semaphore = None
        self._semaphore_loop = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Concurrency limiter bound to the current event loop.

        A new semaphore is created whenever the running loop changes, so the same
        object can be reused across separate asyncio.run() calls.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._semaphore is None or (
            loop is not None and loop is not self._semaphore_loop
        ):
            self._semaphore = asyncio.Semaphore(self._concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    def reset_progress(self, total_tasks: int) -> None:
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.previous_completed_tasks = 0

    async def track_progress(
        self, tasks: list[asyncio.Task], callbacks: list[ProgressBatchCallback]
    ):
        while not all(task.done() for task in tasks):
            await asyncio.sleep(0.1)
            if self.completed_tasks != self.previous_completed_tasks:
                for callback in callbacks:
                    callback.on_batch_change(self.completed_tasks, self.total_tasks)
                self.previous_completed_tasks = self.completed_tasks

        if self.completed_tasks != self.previous_completed_tasks:
            for callback in callbacks:
                callback.on_batch_change(self.completed_tasks, self.total_tasks)

        if self.completed_tasks == self.total_tasks:
            for callback in callbacks:
                callback.on_batch_change(self.completed_tasks, self.total_tasks)

    def progress_callback(self) -> None:
        self.completed_tasks += 1
