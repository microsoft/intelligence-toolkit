# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import os

from util.SecretsHandler import SecretsHandler

from .defaults import (DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_MODEL,
                       DEFAULT_OPENAI_VERSION, DEFAULT_TEMPERATURE)


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" else value

key = 'openai_secretkey'
openai_type_key = 'openai_typekey'
openai_version_key = 'openai_versionkey'
openai_endpoint_key = 'openai_endpointkey'

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
        self._secrets = SecretsHandler()
        
        """Init method definition."""
        self._api_key = config.get("api_key", self.get_api_key())
        self._model = config.get("model", DEFAULT_LLM_MODEL)
        self._api_base = config.get("api_base", self.get_azure_api_base())
        self._api_version = config.get("api_version", DEFAULT_OPENAI_VERSION)
        self._temperature = config.get("temperature", DEFAULT_TEMPERATURE)
        self._max_tokens = config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS)
        self._api_type = config.get("api_type", self.get_openai_type())


    def is_key_stored(self):
        return self._secrets.get_secret(key) != ''
    
    def get_openai_type(self):
        environ = os.environ['OPENAI_TYPE'] if 'OPENAI_TYPE' in os.environ else "OpenAI"
        secret = self._secrets.get_secret(openai_type_key)
        return secret if len(secret) > 0 else environ

    def get_azure_openai_version(self):
        environ = os.environ['AZURE_OPENAI_VERSION'] if 'AZURE_OPENAI_VERSION' in os.environ else DEFAULT_OPENAI_VERSION
        secret = self._secrets.get_secret(openai_version_key)
        return secret if len(secret) > 0 else environ

    def get_azure_api_base(self):
        environ = os.environ['AZURE_OPENAI_ENDPOINT'] if 'AZURE_OPENAI_ENDPOINT' in os.environ else None
        secret = self._secrets.get_secret(openai_endpoint_key)
        return secret if len(secret) > 0 else environ

    def get_api_key(self):
        secret_key = self._secrets.get_secret(key)
        if secret_key != '':
            return secret_key
        env_key = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else '' 
        return env_key

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
