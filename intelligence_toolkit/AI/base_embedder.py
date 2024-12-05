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

from intelligence_toolkit.AI.base_batch_async import BaseBatchAsync
from intelligence_toolkit.AI.classes import VectorData
from intelligence_toolkit.AI.defaults import (
    DEFAULT_CONCURRENT_COROUTINES,
    DEFAULT_LLM_MAX_TOKENS,
    EMBEDDING_BATCHES_NUMBER,
)
from intelligence_toolkit.AI.utils import get_token_count, hash_text
from intelligence_toolkit.AI.vector_store import VectorStore
from intelligence_toolkit.helpers.constants import CACHE_PATH
from intelligence_toolkit.helpers.decorators import retry_with_backoff
from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback

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
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
        self.check_token_count = check_token_count

    @retry_with_backoff()
    async def embed_one_async(
        self,
        data: VectorData,
        has_callback=False,
    ) -> Any | list[float]:
        async with self.semaphore:
            if not data["hash"]:
                text_hashed = hash_text(data["text"])
                data["hash"] = text_hashed
                if self.check_token_count:
                    try:
                        tokens = get_token_count(data["text"])
                        if tokens > self.max_tokens:
                            text = data["text"][: self.max_tokens]
                            data["text"] = text
                            logger.info("Truncated text to max tokens")
                    except Exception:
                        pass
            try:
                embedding = await asyncio.wait_for(
                    self._generate_embedding_async(data["text"]), timeout=90
                )
                data["additional_details"] = json.dumps(
                    data["additional_details"] if "additional_details" in data else {}
                )
                data["vector"] = embedding
            except Exception as e:
                msg = f"Timeout in embedding generation. {e} Please try again."
                raise Exception(msg)

            if has_callback:
                self.progress_callback()
            return embedding, data

    @retry_with_backoff()
    def embed_store_one(
        self, text: str, cache_data=True, additional_detail: Any = "{}"
    ) -> Any | list[float]:
        cache_data = False  # disable for now
        text_hashed = hash_text(text)
        if cache_data:
            existing_embedding = (
                self.vector_store.search_by_column(text_hashed, "hash")
                if cache_data
                else []
            )
            if len(existing_embedding) > 0:
                return existing_embedding.get("vector")[0]

        # error when local
        if self.check_token_count:
            try:
                tokens = get_token_count(text)
                if tokens > self.max_tokens:
                    text = text[: self.max_tokens]
                    logger.info("Truncated text to max tokens")
            except:
                pass

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
        cache_data = False  # disable for now
        self.total_tasks = len(data)
        final_embeddings = []
        loaded_texts = []
        all_data = []

        for i in range(0, len(data), (EMBEDDING_BATCHES_NUMBER)):
            batch_data = data[i : i + (EMBEDDING_BATCHES_NUMBER)]

            if cache_data:
                hash_all_texts = [hash_text(item["text"]) for item in batch_data]
                existing = self.vector_store.search_by_column(hash_all_texts, "hash")

                if len(existing.get("vector")) > 0 and cache_data:
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
                        loaded_texts.append(item[1])
                        final_embeddings.append(item[2])

            new_items = [
                item for item in batch_data if item["text"] not in loaded_texts
            ]
            if len(new_items) > 0:
                tasks = [
                    asyncio.create_task(self.embed_one_async(item, callbacks))
                    for item in new_items
                ]
                if callbacks:
                    progress_task = asyncio.create_task(
                        self.track_progress(tasks, callbacks)
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
