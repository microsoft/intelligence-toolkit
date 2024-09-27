# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from .defaults import (
    DEFAULT_AZ_AUTH_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_OPENAI_VERSION,
    DEFAULT_TEMPERATURE,
)


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" else value


class OpenAIConfiguration:
    """OpenAI Configuration class definition."""

    # Core Configuration
    _api_key: str
    _model: str

    _api_base: str | None
    _api_version: str | None

    _temperature: float | None
    _max_tokens: int | None
    _api_type: str
    _az_auth_type: str
    _embedding_model: str

    def __init__(
        self,
        config: dict | None = None,
    ):
        """Init method definition."""
        if config is None:
            config = {}
        oai_type = self._get_openai_type()
        self._api_key = config.get("api_key", self._get_api_key())
        self._model = config.get(
            "model",
            self._get_chat_model(),
        )
        self._api_base = config.get("api_base", self._get_azure_api_base())
        self._api_version = config.get("api_version", self._get_azure_openai_version())
        self._temperature = config.get("temperature", DEFAULT_TEMPERATURE)
        self._max_tokens = config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS)
        self._az_auth_type = config.get("az_auth_type", self._get_az_auth_type())
        self._api_type = config.get("api_type", oai_type)
        self._embedding_model = config.get(
            "embedding_model", self._get_embedding_model()
        )

    def _get_openai_type(self):
        return os.environ.get("OPENAI_TYPE", "OpenAI")

    def _get_az_auth_type(self):
        return os.environ.get("AZURE_AUTH_TYPE", DEFAULT_AZ_AUTH_TYPE)

    def _get_azure_openai_version(self):
        return os.environ.get("AZURE_OPENAI_VERSION", DEFAULT_OPENAI_VERSION)

    def _get_chat_model(self):
        return os.environ.get("OPENAI_API_MODEL", DEFAULT_LLM_MODEL)

    def _get_embedding_model(self):
        return os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    def _get_azure_api_base(self):
        return os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    def _get_api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def api_key(self) -> str:
        """API key property definition."""
        return self._api_key

    @property
    def model(self) -> str:
        """Model property definition."""
        return self._model

    @property
    def api_base(self) -> str | None:
        """API base property definition."""
        result = _non_blank(self._api_base)
        # Remove trailing slash
        return result[:-1] if result and result.endswith("/") else result

    @property
    def api_version(self) -> str | None:
        """API version property definition."""
        return _non_blank(self._api_version)

    @property
    def temperature(self) -> float | None:
        """Temperature property definition."""
        return self._temperature

    @property
    def max_tokens(self) -> int | None:
        """Max tokens property definition."""
        return self._max_tokens

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @property
    def api_type(self) -> str | None:
        """Type of the AI connection."""
        return self._api_type

    @property
    def az_auth_type(self) -> str:
        """Type of the Azure OpenAI connection."""
        return self._az_auth_type
