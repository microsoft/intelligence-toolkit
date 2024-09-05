# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pyarrow as pa
from tqdm.asyncio import tqdm_asyncio

from toolkit.AI.classes import VectorData
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
        pa.field("additional_details", pa.string()),
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
        data: VectorData,
        callbacks: list[ProgressBatchCallback] | None = None,
    ) -> Any | list[float]:
        async with self.semaphore:
            if not data["hash"]:
                text_hashed = hash_text(data["text"])
                data["hash"] = text_hashed
            tokens = get_token_count(data["text"])
            if tokens > self.max_tokens:
                text = data["text"][: self.max_tokens]
                data["text"] = text
                logger.info("Truncated text to max tokens")
            try:
                embedding = await self._generate_embedding_async(data["text"])
                data["additional_details"] = json.dumps(data["additional_details"])
                data["vector"] = embedding
            except Exception as e:
                msg = f"Problem in embedding generation. {e}"
                raise Exception(msg)

            if callbacks:
                self._progress_callback()
            return embedding, data

    @retry_with_backoff()
    def embed_store_one(
        self, text: str, cache_data=True, additional_detail: Any = "{}"
    ) -> Any | list[float]:
        text_hashed = hash_text(text)
        existing_embedding = (
            self.vector_store.search_by_column(text_hashed, "hash")
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
            data = {
                "hash": text_hashed,
                "text": text,
                "vector": embedding,
                "additional_details": json.dumps(additional_detail),
            }
            self.vector_store.save([data]) if cache_data else None
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise Exception(msg)
        return embedding

    @retry_with_backoff()
    async def embed_store_many(
        self,
        data: list[VectorData],
        callbacks: list[ProgressBatchCallback] | None = None,
        cache_data=True,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        self.total_sentences = len(data)
        final_embeddings = []
        loaded_texts = []
        all_data = []

        for i in range(0, len(data), (EMBEDDING_BATCHES_NUMBER)):
            batch_data = data[i : i + (EMBEDDING_BATCHES_NUMBER)]

            hash_all_texts = [hash_text(item["text"]) for item in batch_data]
            existing = self.vector_store.search_by_column(hash_all_texts, "hash")

            if len(existing.get("vector")) > 0:
                existing_texts = existing.sort_values("text")
                for item in existing_texts.to_numpy():
                    all_data.append(
                        {
                            "hash": item[0],
                            "text": item[1],
                            "vector": item[2],
                            "additional_details": item[3] if len(item) > 3 else {},
                        }
                    )
                    loaded_texts.append(item[1])
                    final_embeddings.append(item[2])

            new_items = [
                item for item in batch_data if item["text"] not in loaded_texts
            ]

            if len(new_items) > 0:
                tasks = [
                    asyncio.create_task(self.embed_one_async(item))
                    for item in new_items
                ]
                if callbacks:
                    progress_task = asyncio.create_task(
                        self._track_progress(tasks, callbacks)
                    )
                result = await tqdm_asyncio.gather(*tasks)
                if callbacks:
                    await progress_task

                embeddings = [embedding[0] for embedding in result]
                new_data = [embedding[1] for embedding in result]
                all_data.extend(new_data)

                final_embeddings.extend(embeddings)
                if cache_data:
                    self.vector_store.save(new_data)
                    self.vector_store.update_duckdb_data()

        print(f"Got {len(loaded_texts)} existing texts")
        logger.info("Got %s existing texts", len(loaded_texts))
        print(f"Got {len(final_embeddings) - len(loaded_texts)} new texts")
        logger.info("Got %s new texts", len(final_embeddings) - len(loaded_texts))

        return all_data

    @abstractmethod
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text"""

    @abstractmethod
    async def _generate_embedding_async(self, text: str) -> list:
        """Generate async embeddings for text"""
