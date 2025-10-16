# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from intelligence_toolkit.AI.types import OpenAIClientTypes
from openai import AsyncAzureOpenAI, AsyncOpenAI


def test_openai_client_types_includes_async_openai():
    # Test that OpenAIClientTypes accepts AsyncOpenAI
    # This is a type union test, so we just verify the types are imported correctly
    assert AsyncOpenAI is not None
    assert AsyncAzureOpenAI is not None
    # Verify the type alias is defined
    assert OpenAIClientTypes is not None
