# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Fixtures: a small multi-period case frame and a fake embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from itk_patterns import DetectionConfig, PatternConfig, PatternDetector
from itk_patterns_testing import FakeEmbedder, cases_frame

if TYPE_CHECKING:
    import polars as pl
    from itk_patterns.models import PatternResult


@pytest.fixture()
def cases() -> pl.DataFrame:
    return cases_frame()


@pytest.fixture()
def config() -> PatternConfig:
    return PatternConfig(period_column="period")


@pytest.fixture()
def detection() -> DetectionConfig:
    return DetectionConfig(min_pattern_count=2, max_pattern_length=3)


@pytest.fixture()
def prepared(cases: pl.DataFrame, config: PatternConfig) -> PatternResult:
    return PatternDetector.prepare(cases, config)


@pytest.fixture()
def fake(prepared: PatternResult) -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def embedded(prepared: PatternResult, fake: FakeEmbedder) -> PatternResult:
    return PatternDetector(fake).embed(prepared)


@pytest.fixture()
def detected(embedded: PatternResult, detection: DetectionConfig) -> PatternResult:
    return PatternDetector.detect(embedded, detection)
