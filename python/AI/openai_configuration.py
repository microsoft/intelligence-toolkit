# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from .defaults import (
    DEFAULT_AZURE_LLM_MODEL,
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

class OpenAIConfiguration():
    """OpenAI Configuration class definition."""

    # Core Configuration
    _api_key: str
    _model: str

    _api_base: str | None
    _api_version: str | None

    _temperature: float | None
    _max_tokens: int | None
    _api_type: str

    def __init__(
        self,
        config: dict = {},
    ):
        """Init method definition."""
        oai_type = self._get_openai_type()
        self._api_key = config.get("api_key", self._get_api_key())
        self._model = config.get("model", DEFAULT_LLM_MODEL if oai_type == 'OpenAI' else self._get_azure_openai_model())
        self._api_base = config.get("api_base", self._get_azure_api_base())
        self._api_version = config.get("api_version", self._get_azure_openai_version())
        self._temperature = config.get("temperature", DEFAULT_TEMPERATURE)
        self._max_tokens = config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS)
        self._api_type = config.get("api_type", oai_type)


    def _get_openai_type(self):
        return os.environ['OPENAI_TYPE'] if 'OPENAI_TYPE' in os.environ else "OpenAI"

    def _get_azure_openai_version(self):
        return os.environ['AZURE_OPENAI_VERSION'] if 'AZURE_OPENAI_VERSION' in os.environ else DEFAULT_OPENAI_VERSION
    
    def _get_azure_openai_model(self):
        return os.environ['AZURE_OPENAI_MODEL'] if 'AZURE_OPENAI_MODEL' in os.environ else DEFAULT_AZURE_LLM_MODEL

    def _get_azure_api_base(self):
        return os.environ['AZURE_OPENAI_ENDPOINT'] if 'AZURE_OPENAI_ENDPOINT' in os.environ else None

    def _get_api_key(self):
        return os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else '' 

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
    def api_type(self) -> str | None:
        """Type of the AI connection."""
        return self._api_type
