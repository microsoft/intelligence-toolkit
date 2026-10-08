# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import asyncio

from tqdm.asyncio import tqdm_asyncio

from intelligence_toolkit.AI.base_batch_async import BaseBatchAsync
from intelligence_toolkit.AI.client import OpenAIClient
from intelligence_toolkit.AI.defaults import DEFAULT_CONCURRENT_COROUTINES
from intelligence_toolkit.AI.errors import LLMError
from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback


class BaseChat(BaseBatchAsync, OpenAIClient):
    def __init__(
        self, configuration=None, concurrent_coroutines=DEFAULT_CONCURRENT_COROUTINES
    ) -> None:
        OpenAIClient.__init__(self, configuration)
        self.set_concurrency(concurrent_coroutines)

    async def generate_text_async(self, messages, callbacks, stream, **llm_kwargs):
        # Transient failures are retried inside OpenAIClient.generate_chat_async.
        async with self.semaphore:
            try:
                chat = await self.generate_chat_async(
                    messages=messages, stream=stream, **llm_kwargs
                )
            except LLMError:
                raise
            except Exception as e:
                msg = f"Problem in OpenAI response. {e}"
                raise LLMError(msg) from e
            if callbacks:
                self.progress_callback()
            return chat

    async def generate_texts_async(
        self,
        messages_list: list[list[dict[str, str]]],
        callbacks: list[ProgressBatchCallback] | None = None,
        **llm_kwargs,
    ):
        self.reset_progress(len(messages_list))
        tasks = [
            asyncio.create_task(
                self.generate_text_async(messages, callbacks, False, **llm_kwargs)
            )
            for messages in messages_list
        ]
        progress_task = (
            asyncio.create_task(self.track_progress(tasks, callbacks))
            if callbacks
            else None
        )
        try:
            return await tqdm_asyncio.gather(*tasks)
        except BaseException:
            # Don't leave sibling requests running (and billing) after a failure.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if progress_task is not None:
                await progress_task
