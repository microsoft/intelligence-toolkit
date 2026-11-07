# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Shared fixtures: a small people/events schema and a deterministic fake chat."""

from __future__ import annotations

import json
from typing import Any

import pytest
from itk_extract.models import Chunk, Extraction
from itk_extract.schema import RecordSchema

PEOPLE_SCHEMA: dict[str, Any] = {
    "title": "People",
    "type": "object",
    "properties": {
        "organisation": {"type": "string"},
        "founded": {"type": ["integer", "null"]},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "contact": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                    },
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


class FakeBatchChat:
    """Stands in for BaseChat.generate_texts_async.

    The user message of each prompt carries the chunk text. When that text is a
    JSON object it is echoed back as the reply, so tests can drive the model's
    answer by writing JSON as the input. Anything else gets ``fallback``.
    """

    def __init__(self, fallback: str = "not json at all") -> None:
        self.fallback = fallback
        self.calls: list[list[list[dict[str, str]]]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def __call__(
        self,
        messages_list: list[list[dict[str, str]]],
        callbacks: Any = None,
        **llm_kwargs: Any,
    ) -> list[str]:
        self.calls.append(messages_list)
        self.kwargs.append(llm_kwargs)
        replies = []
        for messages in messages_list:
            text = messages[-1]["content"].split("--Input text--", 1)[-1].strip()
            try:
                json.loads(text)
            except json.JSONDecodeError:
                replies.append(self.fallback)
            else:
                replies.append(f"```json\n{text}\n```")
        return replies


def chunk(cid: str, text: str = "", source_id: str | None = None) -> Chunk:
    return Chunk(id=cid, source_id=source_id or cid.split("#")[0], index=0, text=text)


def extraction(cid: str, data: dict[str, Any]) -> Extraction:
    return Extraction(chunk(cid), data)


@pytest.fixture()
def schema() -> RecordSchema:
    return RecordSchema(PEOPLE_SCHEMA)


@pytest.fixture()
def fake_chat() -> FakeBatchChat:
    return FakeBatchChat()


@pytest.fixture()
def alice_text() -> str:
    return json.dumps({
        "organisation": "Acme",
        "founded": 1999,
        "people": [
            {
                "name": "Alice",
                "role": "CEO",
                "contact": {"email": "alice@acme.test"},
                "events": [{"date": "2024-01-02", "description": "Appointed CEO"}],
            }
        ],
    })


@pytest.fixture()
def bob_text() -> str:
    return json.dumps({
        "organisation": "ACME Ltd",
        "founded": 1999,
        "people": [
            {
                "name": "Bob",
                "role": "CTO",
                "contact": {"email": "bob@acme.test"},
                "events": [
                    {"date": "2023-05-06", "description": "Joined"},
                    {"date": "2024-02-03", "description": "Promoted"},
                ],
            }
        ],
    })
