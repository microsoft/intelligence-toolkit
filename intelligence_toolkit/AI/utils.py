# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

"""Utility functions for the OpenAI API."""

import hashlib
import json
import logging
from typing import Any

import tiktoken

from intelligence_toolkit.AI.defaults import DEFAULT_ENCODING, DEFAULT_REPORT_BATCH_SIZE
from intelligence_toolkit.AI.validation_prompt import GROUNDEDNESS_PROMPT
from intelligence_toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from intelligence_toolkit.AI.base_chat import BaseChat
from intelligence_toolkit.AI.client import OpenAIClient

log = logging.getLogger(__name__)


def generate_text(ai_configuration, messages, **kwargs):
    return OpenAIClient(ai_configuration).generate_chat(
        messages, stream=False, **kwargs
    )


async def generate_text_async(ai_configuration, messages, **kwargs):
    return await OpenAIClient(ai_configuration).generate_chat_async(messages, **kwargs)


async def map_generate_text(
    ai_configuration,
    messages_list,
    callbacks: list[ProgressBatchCallback] | None = None,
    **kwargs,
):
    return await BaseChat(ai_configuration).generate_texts_async(
        messages_list, callbacks, **kwargs
    )


def get_token_count(text: str, encoding=None, model=None) -> int:
    """Function that counts the number of tokens in a string."""
    encoder = tiktoken.get_encoding(encoding or DEFAULT_ENCODING)
    if model:
        encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(json.dumps(text)))


def prepare_messages(
    system_message: str, variables: dict[str, Any], user_message=None
) -> list[dict[str, str]]:
    """Prepare messages for the OpenAI API."""
    messages = [{"role": "system", "content": system_message.format(**variables)}]

    if user_message is not None:
        messages.append({"role": "user", "content": user_message.format(**variables)})
    return messages


def prepare_validation(messages: str, ai_report: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": GROUNDEDNESS_PROMPT.format(
                instructions=messages, report=ai_report
            ),
        }
    ]


def try_parse_json_object(input: str) -> dict:
    """Generate JSON-string output using best-attempt prompting & parsing techniques."""
    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        log.exception("error loading json, json=%s", input)
        raise
    else:
        if not isinstance(result, dict):
            raise TypeError
        return result


def hash_text(text: str) -> str:
    """Function that hashes a string."""
    text = text.replace("\n", " ")
    return hashlib.sha256(text.encode()).hexdigest()


def generate_messages(
    user_prompt, system_prompt, variables, safety_prompt=""
) -> list[dict[str, str]]:
    full_prompt = f"{system_prompt} {user_prompt} {safety_prompt}"

    return prepare_messages(full_prompt, variables)


def generate_batch_messages(
    prompt,
    batch_name,
    batch_value,
    variables: dict | None = None,
    batch_size: int | None = DEFAULT_REPORT_BATCH_SIZE,
) -> list[dict[str, str]]:
    if variables is None:
        variables = {}

    batch_offset = 0
    batch_count_raw = len(batch_value) // batch_size
    batch_count_remaining = len(batch_value) % batch_size
    batch_count = batch_count_raw + 1 if batch_count_remaining != 0 else batch_count_raw
    batch_messages = []

    full_prompt = " ".join(
        [
            prompt["report_prompt"],
            prompt["user_prompt"],
            prompt["safety_prompt"],
        ]
    )
    for _i in range(batch_count):
        batch = batch_value[
            batch_offset : min(batch_offset + batch_size, len(batch_value))
        ]
        batch_offset += batch_size
        batch_variables = dict(variables)
        batch_variables[batch_name] = batch.to_csv()
        batch_messages.append(prepare_messages(full_prompt, batch_variables))

    return batch_messages
