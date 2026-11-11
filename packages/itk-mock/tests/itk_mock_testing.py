# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Test helpers shared by the itk-mock suite.

Lives outside conftest.py (and the tests directory has no __init__.py) so it
cannot collide with the ``tests`` packages of sibling workspace members.
"""

from __future__ import annotations

import json
import re
from typing import Any

PEOPLE_SCHEMA: dict[str, Any] = {
    "title": "People",
    "type": "object",
    "properties": {
        "dataset": {"type": "string"},
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "address": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "orders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"sku": {"type": "string"}},
                            "required": ["sku"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "age", "address", "tags", "orders"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["dataset", "records"],
    "additionalProperties": False,
}

_COUNT = re.compile(r"(?:Output record count|Total records): (\d+)")
_SEED_NAME = re.compile(r"--Seed Record--\s*\{'name': '([^']+)'")


def canned_reply(
    messages: list[dict[str, str]], response_schema: dict[str, Any] | None
) -> str:
    """Deterministic reply derived only from the prompt.

    Record prompts get the requested number of people, named after the seed
    record (or ``unseeded``); text prompts get a sentence quoting the input.
    """
    content = messages[0]["content"]
    if response_schema is None:
        return "Text about " + content.split("--Input text--", 1)[1].strip()
    count = int(_COUNT.search(content)[1])  # type: ignore[index]
    seed = _SEED_NAME.search(content)
    tag = seed[1] if seed else "unseeded"
    records = [
        {
            "name": f"{tag}/{i}",
            "age": 20 + i,
            "address": {"city": "Lakeside"},
            "tags": ["a", "b"],
            "orders": [{"sku": f"{tag}/{i}/order"}],
        }
        for i in range(count)
    ]
    return json.dumps({"dataset": tag, "records": records})


class FakeChat:
    """Stands in for the model: records every call, replies from the prompt."""

    def __init__(self, reply=canned_reply) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append({
            "messages": messages,
            "response_schema": response_schema,
            "temperature": temperature,
        })
        return self.reply(messages, response_schema)

    @property
    def prompts(self) -> list[str]:
        return [call["messages"][0]["content"] for call in self.calls]
