# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import asyncio
from typing import Any

from sentence_transformers import SentenceTransformer

from intelligence_toolkit.AI.base_embedder import BaseEmbedder
from intelligence_toolkit.AI.defaults import (
    DEFAULT_CONCURRENT_COROUTINES,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
)
from intelligence_toolkit.helpers.constants import CACHE_PATH



class LocalEmbedder(BaseEmbedder):
    def __init__(
        self,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
        concurrent_coroutines: int | None = DEFAULT_CONCURRENT_COROUTINES + 100,
        model: str | None = DEFAULT_LOCAL_EMBEDDING_MODEL,
    ):
        super().__init__(db_name, db_path, max_tokens, concurrent_coroutines, False)
        # Use default model if None is passed
        if model is None:
            model = DEFAULT_LOCAL_EMBEDDING_MODEL
        try:
            self.local_client = SentenceTransformer(model)
        except Exception as e:
            raise Exception(f"Failed to load local embedding model '{model}': {e}. Please ensure the model is available or check your internet connection for download.")

    def _generate_embedding(self, text: str | list[str]) -> list | Any:
        return self.local_client.encode(text).tolist()

    async def _generate_embedding_async(self, text: str) -> list | Any:
        await asyncio.sleep(0)

        return self._generate_embedding(text)