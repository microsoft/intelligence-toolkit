# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import logging

from toolkit.helpers.progress_batch_callback import ProgressBatchCallback

logger = logging.getLogger(__name__)


class BaseBatchAsync:
    total_tasks: int = 1
    completed_tasks: int = 0
    previous_completed_tasks: int = 0

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
