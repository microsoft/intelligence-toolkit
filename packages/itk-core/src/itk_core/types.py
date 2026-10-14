# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

"""A base class for OpenAI-based LLMs."""

from openai import AsyncAzureOpenAI, AsyncOpenAI

OpenAIClientTypes = AsyncOpenAI | AsyncAzureOpenAI
