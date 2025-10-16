# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from unittest.mock import patch

import pytest

from intelligence_toolkit.AI.openai_configuration import OpenAIConfiguration
from intelligence_toolkit.helpers.classes import IntelligenceWorkflow


@pytest.fixture
def openai_config():
    return OpenAIConfiguration({
        "api_key": "test_key",
        "model": "gpt-4",
        "api_type": "OpenAI",
    })


def test_intelligence_workflow_initialization_without_config():
    workflow = IntelligenceWorkflow()
    assert workflow.ai_configuration is None
    assert workflow.embedder is None
    assert workflow.cache_embeddings is True


def test_intelligence_workflow_initialization_with_config(openai_config):
    workflow = IntelligenceWorkflow(openai_config)
    assert workflow.ai_configuration == openai_config
    assert workflow.embedder is None
    assert workflow.cache_embeddings is True


def test_set_ai_configuration(openai_config):
    workflow = IntelligenceWorkflow()
    
    with patch.object(workflow, 'set_embedder') as mock_set_embedder:
        workflow.set_ai_configuration(openai_config)
        
        assert workflow.ai_configuration == openai_config
        mock_set_embedder.assert_called_once()


def test_set_ai_configuration_with_existing_embedder(openai_config):
    workflow = IntelligenceWorkflow()
    workflow.embedder = "mock_embedder"
    
    with patch.object(workflow, 'set_embedder') as mock_set_embedder:
        workflow.set_ai_configuration(openai_config)
        
        assert workflow.ai_configuration == openai_config
        # Should not call set_embedder if embedder already exists
        mock_set_embedder.assert_not_called()


def test_set_embedder_local(openai_config):
    with patch("intelligence_toolkit.helpers.classes.LocalEmbedder") as mock_local:
        workflow = IntelligenceWorkflow(openai_config)
        workflow.set_embedder(local_embedding=True, cache_embeddings=False)
        
        mock_local.assert_called_once()
        assert workflow.embedder is not None
        assert workflow.cache_embeddings is False


def test_set_embedder_openai(openai_config):
    with patch("intelligence_toolkit.helpers.classes.OpenAIEmbedder") as mock_openai, \
         patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        workflow = IntelligenceWorkflow(openai_config)
        workflow.set_embedder(local_embedding=False, cache_embeddings=True)
        
        mock_openai.assert_called_once_with(openai_config)
        assert workflow.embedder is not None
        assert workflow.cache_embeddings is True


def test_set_embedder_default_parameters(openai_config):
    with patch("intelligence_toolkit.helpers.classes.OpenAIEmbedder") as mock_openai, \
         patch("intelligence_toolkit.AI.client.OpenAI"), \
         patch("intelligence_toolkit.AI.client.AsyncOpenAI"):
        
        workflow = IntelligenceWorkflow(openai_config)
        workflow.set_embedder()
        
        # Default is OpenAI embedder with cache_embeddings=True
        mock_openai.assert_called_once()
        assert workflow.cache_embeddings is True
