# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from typing import Any

from sentence_transformers import SentenceTransformer

from python.AI.base_embedder import BaseEmbedder
from python.AI.defaults import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LOCAL_EMBEDDING_MODEL
from python.helpers.constants import CACHE_PATH


class LocalEmbedder(BaseEmbedder):
    def __init__(
        self,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
    ):
        super().__init__(db_name, db_path, max_tokens)
        self.local_client = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)

    def _generate_embedding(self, text: str) -> list | Any:
        return self.local_client.encode(text).tolist()

    def _generate_embeddings(self, texts: list[str]) -> list | Any:
        return self._generate_embedding(texts)
