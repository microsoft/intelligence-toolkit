# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

class LLMCallback:
    """Class for LLM callbacks."""

    def __init__(self):
        self.response = []

    def on_llm_new_token(self, token: str):
        """Handle when a new token is generated."""
        self.response.append(token)

class BatchEmbeddingCallback:
    """Class for LLM callbacks."""
    def __init__(self):
        self.current_batch = 0
        self.total_batches = 0

    def on_embedding_batch_change(self, current: int, total: int):
        """Handle when a new token is generated."""
        self.current_batch = current
        self.total_batches = total