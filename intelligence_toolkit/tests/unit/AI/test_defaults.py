# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from intelligence_toolkit.AI import defaults


def test_default_encoding():
    assert defaults.DEFAULT_ENCODING == "o200k_base"


def test_default_llm_model():
    assert defaults.DEFAULT_LLM_MODEL == "gpt-4.1-mini"


def test_default_llm_max_tokens():
    assert defaults.DEFAULT_LLM_MAX_TOKENS == 4000


def test_default_az_auth_type():
    assert defaults.DEFAULT_AZ_AUTH_TYPE == "Azure Key"


def test_embedding_batches_number():
    assert defaults.EMBEDDING_BATCHES_NUMBER == 500


def test_default_embedding_model():
    assert defaults.DEFAULT_EMBEDDING_MODEL == "text-embedding-3-small"


def test_default_embedding_model_azure():
    assert defaults.DEFAULT_EMBEDDING_MODEL_AZURE == "text-embedding-ada-002"


def test_default_temperature():
    assert defaults.DEFAULT_TEMPERATURE == 0


def test_default_max_input_tokens():
    assert defaults.DEFAULT_MAX_INPUT_TOKENS == 128000


def test_default_openai_version():
    assert defaults.DEFAULT_OPENAI_VERSION == "2024-08-01-preview"


def test_default_local_embedding_model():
    assert defaults.DEFAULT_LOCAL_EMBEDDING_MODEL == "all-distilroberta-v1"


def test_api_base_required_for_azure():
    assert defaults.API_BASE_REQUIRED_FOR_AZURE == "api_base is required for Azure OpenAI client"


def test_chunk_size():
    assert defaults.CHUNK_SIZE == 500


def test_chunk_overlap():
    assert defaults.CHUNK_OVERLAP == 0


def test_default_report_batch_size():
    assert defaults.DEFAULT_REPORT_BATCH_SIZE == 100


def test_default_concurrent_coroutines():
    assert defaults.DEFAULT_CONCURRENT_COROUTINES == 50
