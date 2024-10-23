# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from toolkit.AI.local_embedder import LocalEmbedder
from toolkit.AI.openai_configuration import OpenAIConfiguration
from toolkit.AI.openai_embedder import OpenAIEmbedder


class IntelligenceWorkflow:
    # Base class for all AI workflows
    def __init__(self, ai_configuration: OpenAIConfiguration | None = None) -> None:
        self.ai_configuration = ai_configuration

    def set_ai_configuration(self, ai_configuration: OpenAIConfiguration) -> None:
        self.ai_configuration = ai_configuration
        if not self.embedder:
            self.set_embedder()

    def set_embedder(
        self, local_embedding: bool = False, cache_embeddings: bool = True
    ):
        if local_embedding:
            self.embedder = LocalEmbedder()
        else:
            self.embedder = OpenAIEmbedder(self.ai_configuration)
        self.cache_embeddings = cache_embeddings