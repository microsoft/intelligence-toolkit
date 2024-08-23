# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import logging

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI

from .classes import LLMCallback
from .defaults import API_BASE_REQUIRED_FOR_AZURE, DEFAULT_EMBEDDING_MODEL
from .openai_configuration import OpenAIConfiguration

log = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI Client class definition."""

    _client = None

    def __init__(self, configuration: OpenAIConfiguration | None = None) -> None:
        self.configuration = configuration or OpenAIConfiguration()
        self.create_openai_client()
        self.semaphore = asyncio.Semaphore(5)

    def create_openai_client(self) -> None:
        """Create a new OpenAI client instance."""
        if self.configuration.api_type == "Azure OpenAI":
            api_base = self.configuration.api_base
            if api_base is None:
                raise ValueError(API_BASE_REQUIRED_FOR_AZURE)
            log.info(
                "Creating Azure OpenAI client api_base=%s",
                api_base,
            )

            if self.configuration.az_auth_type == "Managed Identity":
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )

                self._client = AzureOpenAI(
                    api_version=self.configuration.api_version,
                    # Azure-Specifics
                    azure_ad_token_provider=token_provider,
                    azure_endpoint=api_base,
                )
            else:
                self._client = AzureOpenAI(
                    api_version=self.configuration.api_version,
                    # Azure-Specifics
                    azure_endpoint=api_base,
                    api_key=self.configuration.api_key,
                )
        else:
            log.info("Creating OpenAI client")
            self._client = OpenAI(
                api_key=self.configuration.api_key,
            )
        return self._client

    def generate_chat(
        self,
        messages: list[str],
        stream: bool = True,
        callbacks: list[LLMCallback] | None = None,
        **kwargs,
    ):
        try:
            if "max_tokens" in kwargs.keys():
                max_tokens = kwargs["max_tokens"]
                kwargs.pop("max_tokens")
            else:
                max_tokens = self.configuration.max_tokens
            response = self._client.chat.completions.create(
                model=self.configuration.model,
                temperature=self.configuration.temperature,
                max_tokens=max_tokens,
                messages=messages,
                stream=stream,
                **kwargs,
            )

            if stream and callbacks is not None:
                full_response = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""  # type: ignore
                    if delta is not None:
                        full_response += delta
                        if callbacks:
                            show = full_response
                            if len(delta) > 0:
                                show += "▌"
                            for callback in callbacks:
                                callback.on_llm_new_token(show)
                return full_response

            return response.choices[0].message.content or ""  # type: ignore
        except Exception as e:
            print(f"Error validating report: {e}")
            msg = f"Problem in OpenAI response. {e}"
            raise Exception(msg)

    async def agenerate_chat(
        self,
        messages: list[str],
        callbacks: list[LLMCallback] | None = None,
        **kwargs,
    ):
        try:
            if "max_tokens" in kwargs.keys():
                max_tokens = kwargs["max_tokens"]
                kwargs.pop("max_tokens")
            else:
                max_tokens = self.configuration.max_tokens
            if "stream" in kwargs.keys():
                kwargs.pop("stream")
            response = self._client.chat.completions.create(
                model=self.configuration.model,
                temperature=self.configuration.temperature,
                max_tokens=max_tokens,
                messages=messages,
                stream=False,
                **kwargs,
            )
            return response.choices[0].message.content or ""  # type: ignore
        except Exception as e:
            print(f"Error validating report: {e}")
            msg = f"Problem in OpenAI response. {e}"
            raise Exception(msg) from e

    def generate_embedding(
        self, text: str, model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        embedding = self._client.embeddings.create(input=text, model=model)
        return embedding.data[0].embedding

    def generate_embeddings(
        self, text: list[str], model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        return self._client.embeddings.create(input=text, model=model)

    async def map_generate_text(
        self,
        messages_list: list[list[dict[str, str]]],
        **llm_kwargs,
    ):
        return await asyncio.gather(
            *[self.agenerate_chat(messages, **llm_kwargs) for messages in messages_list]
        )
