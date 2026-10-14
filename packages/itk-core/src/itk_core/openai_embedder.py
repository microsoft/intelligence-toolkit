# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from itk_core.base_embedder import BaseEmbedder
from itk_core.client import OpenAIClient
from itk_core.defaults import DEFAULT_CONCURRENT_COROUTINES
from itk_core.openai_configuration import OpenAIConfiguration
from itk_core.paths import CACHE_PATH


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        configuration: OpenAIConfiguration,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
        concurrent_coroutines: int | None = DEFAULT_CONCURRENT_COROUTINES,
    ):
        super().__init__(
            db_name, db_path, configuration.max_tokens, concurrent_coroutines
        )
        self.configuration = configuration
        self.openai_client = OpenAIClient(configuration)

    def _generate_embedding(self, text: str) -> list[float]:
        return self.openai_client.generate_embedding(
            text, model=self.configuration.embedding_model
        )

    async def _generate_embedding_async(self, text: str) -> list[float]:
        return await self.openai_client.generate_embedding_async(
            text, model=self.configuration.embedding_model
        )