# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pyarrow as pa

from toolkit.AI.defaults import DEFAULT_LLM_MAX_TOKENS
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
    vector_store: VectorStore
    max_tokens: int

    def __init__(
        self,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        self.vector_store = VectorStore(db_name, db_path, schema)
        self.max_tokens = max_tokens

    @retry_with_backoff()
    async def embed_store_many_async(
        self, texts: list[str], cache_data=True
    ) -> Any | list[float]:
        tasks = [self.embed_store_one_async(text, cache_data) for text in texts]
        return await asyncio.gather(*tasks)

    async def embed_store_one_async(
        self, text: str, cache_data=True
    ) -> Any | list[float]:
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
            embedding = await self._generate_embedding_async(text)
            data = {"hash": text_hashed, "text": text, "vector": embedding}
            self.vector_store.save([data]) if cache_data else None
        except Exception as e:
            msg = f"Problem in embedding generation. {e}"
            raise Exception(msg)
        return embedding

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
    def embed_store_many(
        self,
        texts: list[str],
        callback: ProgressBatchCallback | None = None,
        cache_data=True,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        final_embeddings = [None] * len(texts)
        new_texts = []
        existing_texts_count = 0

        for ix, text in enumerate(texts):
            text_hashed = hash_text(text)
            existing_embedding = (
                self.vector_store.search_one_by_column(text_hashed, "hash")
                if cache_data
                else []
            )
            if not len(existing_embedding):
                new_texts.append((ix, text))
            else:
                final_embeddings[ix] = existing_embedding.get("vector")[0]
                existing_texts_count += 1

        print(f"Got {existing_texts_count} existing texts")
        logger.info("Got %s existing texts", existing_texts_count)
        print(f"Got {len(new_texts)} new texts")
        logger.info("Got %s new texts", len(new_texts))

        num_batches = len(new_texts) // 2000 + 1
        batch_count = 1
        loaded_embeddings = []

        for i in range(0, len(new_texts), 2000):
            if callback:
                for cb in callback:
                    cb.on_batch_change(batch_count, num_batches)
            batch_count += 1
            batch = new_texts[i : i + 2000]
            batch_texts = [x[1] for x in batch]

            try:
                embeddings = self._generate_embeddings(batch_texts)
            except Exception as e:
                msg = f"Problem in embedding generation. {e}"
                raise Exception(msg)

            for j, (ix, text) in enumerate(batch):
                text_hashed = hash_text(text)
                loaded_embeddings.append(
                    {"hash": text_hashed, "text": text, "vector": embeddings[j]}
                )
                final_embeddings[ix] = np.array(embeddings[j])

        if loaded_embeddings and cache_data:
            self.vector_store.save(loaded_embeddings) if cache_data else None
            self.vector_store.update_duckdb_data()
        return np.array(final_embeddings)

    @abstractmethod
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text"""

    @abstractmethod
    def _generate_embeddings(self, texts: list[str]) -> list:
        """Generate embeddings for multiple texts"""

    @abstractmethod
    async def _generate_embedding_async(self, text: str) -> list:
        """Generate async embeddings for text"""