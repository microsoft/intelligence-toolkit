# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from python.AI.base_embedder import BaseEmbedder
from python.helpers.constants import CACHE_PATH

from .client import OpenAIClient
from .openai_configuration import OpenAIConfiguration


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        configuration: OpenAIConfiguration,
        db_name: str = "embeddings",
        db_path=CACHE_PATH,
    ):
        super().__init__(db_name, db_path, configuration.max_tokens)
        self.configuration = configuration
        self.openai_client = OpenAIClient(configuration)

    def _generate_embedding(self, text: str) -> list[float]:
        return self.openai_client.generate_embedding(text)

    def _generate_embeddings(self, texts: list[str]) -> list:
        return [x.embedding for x in self.openai_client.generate_embeddings(texts).data]
