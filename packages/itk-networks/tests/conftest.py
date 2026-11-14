# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Fixtures: the sample tables, a prepared result and a detector with fakes."""

from __future__ import annotations

import pytest
from itk_networks.detector import NetworkDetector
from itk_networks.models import NetworkConfig, NetworkResult
from itk_networks_testing import (
    ChunkDetector,
    FakeChat,
    FakeEmbedder,
    attribute_table,
    flag_table,
    group_table,
)


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def chunker() -> ChunkDetector:
    return ChunkDetector()


@pytest.fixture()
def fake_chat() -> FakeChat:
    return FakeChat()


@pytest.fixture()
def detector(
    embedder: FakeEmbedder, chunker: ChunkDetector, fake_chat: FakeChat
) -> NetworkDetector:
    return NetworkDetector(embedder, chunker, fake_chat)


@pytest.fixture()
def prepared() -> NetworkResult:
    return NetworkDetector.prepare([attribute_table()], [flag_table()], [group_table()])


@pytest.fixture()
def config() -> NetworkConfig:
    return NetworkConfig(max_attribute_degree=3)


@pytest.fixture()
def identified(
    detector: NetworkDetector, prepared: NetworkResult, config: NetworkConfig
) -> NetworkResult:
    return detector.identify(prepared, config)
