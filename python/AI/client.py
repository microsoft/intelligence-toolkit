# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import logging
from typing import List

from openai import AzureOpenAI, OpenAI

from .classes import LLMCallback
from .defaults import API_BASE_REQUIRED_FOR_AZURE, DEFAULT_EMBEDDING_MODEL
from .openai_configuration import OpenAIConfiguration

log = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI Client class definition."""
    _client = None
    def __init__(self, configuration: OpenAIConfiguration = OpenAIConfiguration()) -> None:
        self.configuration = configuration
        self.create_openai_client()

    def create_openai_client(self) -> None:
        """Create a new OpenAI client instance."""
        if self.configuration.api_type == 'Azure OpenAI':
            api_base = self.configuration.api_base
            if api_base is None:
                raise ValueError(API_BASE_REQUIRED_FOR_AZURE)

            log.info(
                "Creating Azure OpenAI client api_base=%s",
                api_base,
            )
            self._client = AzureOpenAI(
                api_key=self.configuration.api_key,
                # Azure-Specifics
                api_version=self.configuration.api_version,
                azure_endpoint=api_base,
            )
            return

        log.info("Creating OpenAI client")
        self._client = OpenAI(
            api_key=self.configuration.api_key,
        )

    def generate_chat(self, messages: List[str], stream: bool = True, callbacks: List[LLMCallback] = None):
        try:
            response = self._client.chat.completions.create(
                model=self.configuration.model,
                temperature=self.configuration.temperature,
                max_tokens=self.configuration.max_tokens,
                messages=messages,
                stream=stream,
            )

            if stream:
                full_response = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""  # type: ignore
                    if delta is not None:
                        full_response += delta
                        if callbacks:
                            show = full_response
                            if len(delta) > 0:
                                show += '▌'
                            for callback in callbacks:
                                callback.on_llm_new_token(show)
                return full_response
            
            return response.choices[0].message.content or ""  # type: ignore
        except Exception as e:
            print(f'Error validating report: {e}')
            raise Exception(f'Problem in OpenAI response. {e}')
        
    def generate_embedding(self, text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
        embedding = self._client.embeddings.create(input = text, model=model)
        return embedding.data[0].embedding
    
    def generate_embeddings(self, text: List[str], model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
        return self._client.embeddings.create(input = text, model=model)