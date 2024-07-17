# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

"""Utility functions for the OpenAI API."""

import hashlib
import json
import logging
from typing import Any

import tiktoken

from .defaults import DEFAULT_ENCODING
from .validation_prompt import GROUNDEDNESS_PROMPT

log = logging.getLogger(__name__)


def get_token_count(text: str, encoding=DEFAULT_ENCODING) -> int:
    """Function that counts the number of tokens in a string."""
    encoder = tiktoken.get_encoding(encoding)
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
