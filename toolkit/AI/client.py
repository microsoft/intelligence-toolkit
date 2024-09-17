# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import logging

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from .classes import LLMCallback
from .defaults import API_BASE_REQUIRED_FOR_AZURE, DEFAULT_EMBEDDING_MODEL
from .openai_configuration import OpenAIConfiguration

log = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI Client class definition."""

    _client = None
    _async_client = None

    def __init__(
        self, configuration: OpenAIConfiguration | None = None, concurrent_coroutines=10
    ) -> None:
        self.configuration = configuration or OpenAIConfiguration()
        self._create_openai_client()

    def _create_openai_client(self) -> None:
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
                self._async_client = AsyncAzureOpenAI(
                    api_version=self.configuration.api_version,
                    # Azure-Specifics
                    azure_ad_token_provider=token_provider,
                    azure_endpoint=api_base,
                )
            else:
                self._client = AsyncAzureOpenAI(
                    api_version=self.configuration.api_version,
                    # Azure-Specifics
                    azure_endpoint=api_base,
                    api_key=self.configuration.api_key,
                )
                self._async_client = AzureOpenAI(
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
            self._async_client = AsyncOpenAI(
                api_key=self.configuration.api_key,
            )

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

    async def generate_chat_async(
        self,
        messages: list[str],
        **kwargs,
    ):
        if "max_tokens" in kwargs.keys():
            max_tokens = kwargs["max_tokens"]
            kwargs.pop("max_tokens")
        else:
            max_tokens = self.configuration.max_tokens
        if "temperature" in kwargs.keys():
            temperature = kwargs["temperature"]
            kwargs.pop("temperature")
        else:
            temperature = self.configuration.temperature
        if "stream" in kwargs.keys():
            kwargs.pop("stream")
        response = await self._async_client.chat.completions.create(
            model=self.configuration.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""  # type: ignore

    def generate_embedding(
        self, text: str, model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        embedding = self._client.embeddings.create(input=text, model=model)
        return embedding.data[0].embedding

    def generate_embeddings(
        self, text: list[str], model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        return self._client.embeddings.create(input=text, model=model)

    async def generate_embedding_async(
        self, text: list[str], model: str = DEFAULT_EMBEDDING_MODEL
    ) -> list[float]:
        embedding = await self._async_client.embeddings.create(input=text, model=model)
        return embedding.data[0].embedding