# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import asyncio

from tqdm.asyncio import tqdm_asyncio

from intelligence_toolkit.AI.base_batch_async import BaseBatchAsync
from intelligence_toolkit.AI.client import OpenAIClient
from intelligence_toolkit.AI.defaults import DEFAULT_CONCURRENT_COROUTINES
from intelligence_toolkit.helpers.decorators import retry_with_backoff
from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback


class BaseChat(BaseBatchAsync, OpenAIClient):
    def __init__(
        self, configuration=None, concurrent_coroutines=DEFAULT_CONCURRENT_COROUTINES
    ) -> None:
        OpenAIClient.__init__(self, configuration)
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)

    @retry_with_backoff()
    async def generate_text_async(self, messages, callbacks, **llm_kwargs):
        async with self.semaphore:
            try:
                chat = await self.generate_chat_async(messages, **llm_kwargs)
                if callbacks:
                    self.progress_callback()
            except Exception as e:
                print(f"Error validating report: {e}")
                msg = f"Problem in OpenAI response. {e}"
                raise Exception(msg) from e
            return chat

    @retry_with_backoff()
    async def generate_texts_async(
        self,
        messages_list: list[list[dict[str, str]]],
        callbacks: list[ProgressBatchCallback] | None = None,
        **llm_kwargs,
    ):
        self.total_tasks = len(messages_list)
        tasks = [
            asyncio.create_task(
                self.generate_text_async(messages, callbacks, **llm_kwargs)
            )
            for messages in messages_list
        ]
        if callbacks:
            progress_task = asyncio.create_task(self.track_progress(tasks, callbacks))
        result = await tqdm_asyncio.gather(*tasks)
        if callbacks:
            await progress_task
        return result
