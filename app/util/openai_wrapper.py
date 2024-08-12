# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from toolkit.AI.openai_configuration import OpenAIConfiguration

from .secrets_handler import SecretsHandler

key = "openai_secret"
openai_type_key = "openai_type"
openai_version_key = "openai_version"
openai_endpoint_key = "openai_endpoint"
openai_model_key = "openai_model"
openai_azure_auth_type = "openai_azure_auth_type"


class UIOpenAIConfiguration:
    def __init__(
        self,
    ):
        self._secrets = SecretsHandler()

    def get_configuration(self) -> OpenAIConfiguration:
        api_type = self._secrets.get_secret(openai_type_key) or None
        version = self._secrets.get_secret(openai_version_key) or None
        endpoint = self._secrets.get_secret(openai_endpoint_key) or None
        secret_key = self._secrets.get_secret(key) or None
        model = self._secrets.get_secret(openai_model_key) or None
        az_auth_type = self._secrets.get_secret(openai_azure_auth_type) or None

        config = {
            "api_type": api_type,
            "api_version": version,
            "api_base": endpoint,
            "api_key": secret_key,
            "model": model,
            "az_auth_type": az_auth_type,
        }
        values = {k: v for k, v in config.items() if v is not None}
        return OpenAIConfiguration(values)
