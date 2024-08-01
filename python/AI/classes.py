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
