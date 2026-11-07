# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Fixtures: the people/events schema and a deterministic fake chat."""

from __future__ import annotations

import json

import pytest
from itk_extract.schema import RecordSchema
from itk_extract_testing import PEOPLE_SCHEMA, FakeBatchChat


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
