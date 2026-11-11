# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Fixtures: a people schema, a boilerplate schema and a deterministic fake chat."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from itk_mock.schema import create_boilerplate_schema
from itk_mock_testing import PEOPLE_SCHEMA, FakeChat


@pytest.fixture()
def people_schema() -> dict[str, Any]:
    return copy.deepcopy(PEOPLE_SCHEMA)


@pytest.fixture()
def boilerplate() -> dict[str, Any]:
    return create_boilerplate_schema()


@pytest.fixture()
def fake_chat() -> FakeChat:
    return FakeChat()
