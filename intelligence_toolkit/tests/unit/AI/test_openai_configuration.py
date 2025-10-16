# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
from unittest.mock import patch

import pytest

from intelligence_toolkit.AI.defaults import (
    DEFAULT_AZ_AUTH_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_AZURE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_OPENAI_VERSION,
    DEFAULT_TEMPERATURE,
)
from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all OpenAI-related environment variables for clean testing."""
    env_vars = [
        "OPENAI_TYPE",
        "AZURE_AUTH_TYPE",
        "AZURE_OPENAI_VERSION",
        "OPENAI_API_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_KEY",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


def test_openai_configuration_initialization_defaults(clean_env):
    config = OpenAIConfiguration()
    
    assert config.api_key == ""
    assert config.model == DEFAULT_LLM_MODEL
    assert config.api_base is None  # Empty string becomes None via _non_blank
    assert config.api_version == DEFAULT_OPENAI_VERSION
    assert config.temperature == DEFAULT_TEMPERATURE
    assert config.max_tokens == DEFAULT_LLM_MAX_TOKENS
    assert config.az_auth_type == DEFAULT_AZ_AUTH_TYPE
    assert config.api_type == "OpenAI"
    assert config.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_openai_configuration_initialization_with_config():
    config_dict = {
        "api_key": "test_key_123",
        "model": "gpt-4",
        "api_base": "https://test.openai.azure.com",
        "api_version": "2023-05-15",
        "temperature": 0.7,
        "max_tokens": 2000,
        "az_auth_type": "Managed Identity",
        "api_type": "Azure OpenAI",
        "embedding_model": "text-embedding-ada-002",
    }
    
    config = OpenAIConfiguration(config_dict)
    
    assert config.api_key == "test_key_123"
    assert config.model == "gpt-4"
    assert config.api_base == "https://test.openai.azure.com"
    assert config.api_version == "2023-05-15"
    assert config.temperature == 0.7
    assert config.max_tokens == 2000
    assert config.az_auth_type == "Managed Identity"
    assert config.api_type == "Azure OpenAI"
    assert config.embedding_model == "text-embedding-ada-002"


def test_openai_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_TYPE", "Azure OpenAI")
    monkeypatch.setenv("AZURE_AUTH_TYPE", "Managed Identity")
    monkeypatch.setenv("AZURE_OPENAI_VERSION", "2024-02-01")
    monkeypatch.setenv("OPENAI_API_MODEL", "gpt-4-turbo")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-resource.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_KEY", "env_api_key")
    
    config = OpenAIConfiguration()
    
    assert config.api_type == "Azure OpenAI"
    assert config.az_auth_type == "Managed Identity"
    assert config.api_version == "2024-02-01"
    assert config.model == "gpt-4-turbo"
    assert config.embedding_model == "text-embedding-3-large"
    assert config.api_base == "https://my-resource.openai.azure.com"
    assert config.api_key == "env_api_key"


def test_openai_configuration_api_base_trailing_slash_removal():
    config_dict = {
        "api_base": "https://test.openai.azure.com/",
    }
    
    config = OpenAIConfiguration(config_dict)
    assert config.api_base == "https://test.openai.azure.com"


def test_openai_configuration_api_base_no_trailing_slash():
    config_dict = {
        "api_base": "https://test.openai.azure.com",
    }
    
    config = OpenAIConfiguration(config_dict)
    assert config.api_base == "https://test.openai.azure.com"


def test_openai_configuration_blank_api_base():
    config_dict = {
        "api_base": "   ",
    }
    
    config = OpenAIConfiguration(config_dict)
    assert config.api_base is None


def test_openai_configuration_blank_api_version():
    config_dict = {
        "api_version": "   ",
    }
    
    config = OpenAIConfiguration(config_dict)
    assert config.api_version is None


def test_openai_configuration_azure_embedding_model_default(monkeypatch):
    monkeypatch.setenv("OPENAI_TYPE", "Azure OpenAI")
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    
    config = OpenAIConfiguration()
    assert config.embedding_model == DEFAULT_EMBEDDING_MODEL_AZURE


def test_openai_configuration_openai_embedding_model_default(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_TYPE", "OpenAI")
    
    config = OpenAIConfiguration()
    assert config.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_openai_configuration_properties_are_accessible():
    config = OpenAIConfiguration()
    
    # Test that all properties are accessible
    _ = config.api_key
    _ = config.model
    _ = config.api_base
    _ = config.api_version
    _ = config.temperature
    _ = config.max_tokens
    _ = config.embedding_model
    _ = config.api_type
    _ = config.az_auth_type
