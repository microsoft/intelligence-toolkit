# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pyarrow as pa
from tqdm.asyncio import tqdm_asyncio

from toolkit.AI.defaults import DEFAULT_LLM_MAX_TOKENS, EMBEDDING_BATCHES_NUMBER
from toolkit.AI.vector_store import VectorStore
from toolkit.helpers.constants import CACHE_PATH
from toolkit.helpers.decorators import retry_with_backoff
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback

from .utils import get_token_count, hash_text

logger = logging.getLogger(__name__)

schema = pa.schema(
    [
        pa.field("hash", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float64())),
    ]
)


class BaseEmbedder(ABC):
    def __init__(
        self,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
        concurrent_coroutines=100,
    ) -> None:
        self.vector_store = VectorStore(db_name, db_path, schema)
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
        self.total_sentences: int = 1
        self.completed_tasks: int = 0
        self.previous_completed_tasks: int = 0

    async def _track_progress(
        self, tasks: list[asyncio.Task], callbacks: list[ProgressBatchCallback]
    ):
        while not all(task.done() for task in tasks):
            await asyncio.sleep(0.1)
            if self.completed_tasks != self.previous_completed_tasks:
                for callback in callbacks:
                    callback.on_batch_change(self.completed_tasks, self.total_sentences)
                self.previous_completed_tasks = self.completed_tasks
        # Ensure final update
        if self.completed_tasks != self.previous_completed_tasks:
            for callback in callbacks:
                callback.on_batch_change(self.completed_tasks, self.total_sentences)

    def _progress_callback(self):
        self.completed_tasks += 1

    @retry_with_backoff()
    async def embed_one_async(
        self,
        text: str,
        callbacks: list[ProgressBatchCallback] | None = None,
    ) -> Any | list[float]:
        async with self.semaphore:
            text_hashed = hash_text(text)
            tokens = get_token_count(text)
            if tokens > self.max_tokens:
                text = text[: self.max_tokens]
                logger.info("Truncated text to max tokens")

            try:
                embedding = await self._generate_embedding_async(text)
                data = {"hash": text_hashed, "text": text, "vector": embedding}
            except Exception as e:
                msg = f"Problem in embedding generation. {e}"
                raise Exception(msg)

            if len(callbacks) > 0:
                self._progress_callback()
            return embedding, data

    @retry_with_backoff()
    def embed_store_one(self, text: str, cache_data=True) -> Any | list[float]:
        text_hashed = hash_text(text)
        existing_embedding = (
            self.vector_store.search_one_by_column(text_hashed, "hash")
            if cache_data
            else []
        )
        if len(existing_embedding) > 0:
            return existing_embedding.get("vector")[0]

        tokens = get_token_count(text)
        if tokens > self.max_tokens:
            text = text[: self.max_tokens]
            logger.info("Truncated text to max tokens")

        try:
            embedding = self._generate_embedding(text)
            data = {"hash": text_hashed, "text": text, "vector": embedding}
            self.vector_store.save([data]) if cache_data else None
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise Exception(msg)
        return embedding

    @retry_with_backoff()
    async def embed_store_many(
        self,
        texts: list[str],
        callbacks: list[ProgressBatchCallback] | None = None,
        cache_data=True,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        self.total_sentences = len(texts)
        final_embeddings = []
        loaded_texts = []

        # for each batch
        for i in range(0, len(texts), (EMBEDDING_BATCHES_NUMBER)):
            batch_texts = texts[i : i + (EMBEDDING_BATCHES_NUMBER)]
            # texts from this batch
            hash_all_texts = [hash_text(te) for te in batch_texts]
            existing = self.vector_store.search_by_column(hash_all_texts, "hash")
            if len(existing.get("vector")) > 0:
                existing_texts = existing.sort_values("text")
                for text in existing_texts.to_numpy():
                    loaded_texts.append(text[1])
                    final_embeddings.append(text[2])

            new_texts = list(set(batch_texts) - set(loaded_texts))

            if len(new_texts) > 0:
                tasks = [
                    asyncio.create_task(self.embed_one_async(text, []))
                    for text in new_texts
                ]
                if callbacks:
                    progress_task = asyncio.create_task(
                        self._track_progress(tasks, callbacks)
                    )
                result = await tqdm_asyncio.gather(*tasks)
                if callbacks:
                    await progress_task

                embeddings = [embedding[0] for embedding in result]
                data = [embedding[1] for embedding in result]

                final_embeddings.extend(embeddings)
                if cache_data:
                    self.vector_store.save(data)
                    self.vector_store.update_duckdb_data()

        print(f"Got {len(loaded_texts)} existing texts")
        logger.info("Got %s existing texts", len(loaded_texts))
        print(f"Got {len(final_embeddings) - len(loaded_texts)} new texts")
        logger.info("Got %s new texts", len(final_embeddings) - len(loaded_texts))

        return np.array(final_embeddings)

    @abstractmethod
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text"""

    @abstractmethod
    async def _generate_embedding_async(self, text: str) -> list:
        """Generate async embeddings for text"""
