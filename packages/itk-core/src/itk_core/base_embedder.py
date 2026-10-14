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

from itk_core.base_batch_async import BaseBatchAsync
from itk_core.classes import VectorData
from itk_core.defaults import (
    DEFAULT_CONCURRENT_COROUTINES,
    DEFAULT_LLM_MAX_TOKENS,
    EMBEDDING_BATCHES_NUMBER,
)
from itk_core.errors import EmbeddingError
from itk_core.paths import CACHE_PATH
from itk_core.progress import ProgressBatchCallback
from itk_core.utils import hash_text, truncate_to_tokens
from itk_core.vector_store import VectorStore

logger = logging.getLogger(__name__)

schema = pa.schema(
    [
        pa.field("hash", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float64())),
        pa.field("additional_details", pa.string()),
    ]
)


class BaseEmbedder(ABC, BaseBatchAsync):
    """Batch embedding with concurrency limiting and progress tracking.

    Transient API failures are retried by the underlying client (see
    OpenAIClient); this class does not add another retry layer.
    """

    def __init__(
        self,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
        concurrent_coroutines=DEFAULT_CONCURRENT_COROUTINES,
        check_token_count=True,
    ) -> None:
        self.vector_store = VectorStore(db_name, db_path, schema)
        self.max_tokens = max_tokens
        self.set_concurrency(concurrent_coroutines)
        self.check_token_count = check_token_count

    def _truncate(self, text: str) -> str:
        if not self.check_token_count:
            return text
        try:
            truncated = truncate_to_tokens(text, self.max_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning("Token counting failed, embedding untruncated text: %s", e)
            return text
        if truncated is not text:
            logger.info("Truncated text to %s tokens", self.max_tokens)
        return truncated

    async def embed_one_async(
        self,
        data: VectorData,
        has_callback=False,
    ) -> Any | list[float]:
        async with self.semaphore:
            if not data["hash"]:
                data["hash"] = hash_text(data["text"])
                data["text"] = self._truncate(data["text"])
            try:
                embedding = await self._generate_embedding_async(data["text"])
            except Exception as e:
                msg = f"Problem in embedding generation. {e}"
                raise EmbeddingError(msg) from e
            data["additional_details"] = json.dumps(
                data["additional_details"] if "additional_details" in data else {}
            )
            data["vector"] = embedding

            if has_callback:
                self.progress_callback()
            return embedding, data

    def embed_store_one(
        self, text: str, cache_data=True, additional_detail: Any = "{}"
    ) -> Any | list[float]:
        cache_data = False  # disable for now
        text_hashed = hash_text(text)
        if cache_data:
            existing_embedding = self.vector_store.search_by_column(text_hashed, "hash")
            if len(existing_embedding) > 0:
                return existing_embedding.get("vector")[0]

        text = self._truncate(text)
        try:
            embedding = self._generate_embedding(text)
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise EmbeddingError(msg) from e
        if cache_data:
            self.vector_store.save(
                [
                    {
                        "hash": text_hashed,
                        "text": text,
                        "vector": embedding,
                        "additional_details": json.dumps(additional_detail),
                    }
                ]
            )
        return embedding

    async def _gather_batch(
        self,
        items: list[VectorData],
        callbacks: list[ProgressBatchCallback] | None,
    ) -> list:
        tasks = [
            asyncio.create_task(self.embed_one_async(item, bool(callbacks)))
            for item in items
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

    async def embed_store_many(
        self,
        data: list[VectorData],
        callbacks: list[ProgressBatchCallback] | None = None,
        cache_data=True,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        cache_data = False  # disable for now
        self.reset_progress(len(data))
        final_embeddings = []
        loaded_texts = set()
        all_data = []

        for i in range(0, len(data), EMBEDDING_BATCHES_NUMBER):
            batch_data = data[i : i + EMBEDDING_BATCHES_NUMBER]

            if cache_data:
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
                                "additional_details": item[3]
                                if len(item) > 3
                                else "{}",
                            }
                        )
                        loaded_texts.add(item[1])
                        final_embeddings.append(item[2])

            new_items = [
                item for item in batch_data if item["text"] not in loaded_texts
            ]
            if len(new_items) > 0:
                result = await self._gather_batch(new_items, callbacks)
                embeddings = [embedding[0] for embedding in result]
                new_data = [embedding[1] for embedding in result]
                all_data.extend(new_data)

                final_embeddings.extend(embeddings)
                if cache_data:
                    self.vector_store.save(new_data)
                    self.vector_store.update_duckdb_data()

        logger.info("Got %s existing texts", len(loaded_texts))
        logger.info("Got %s new texts", len(final_embeddings) - len(loaded_texts))

        return all_data

    @abstractmethod
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text"""

    @abstractmethod
    async def _generate_embedding_async(self, text: str) -> list:
        """Generate async embeddings for text"""
