# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Fixtures: a small complaints frame, configs with and without windows, a fake chat."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from itk_compare import CaseGroupComparer, CompareConfig, CompareResult
from itk_compare_testing import FakeChat, complaints_frame

if TYPE_CHECKING:
    import polars as pl


@pytest.fixture()
def complaints() -> pl.DataFrame:
    return complaints_frame()


@pytest.fixture()
def config() -> CompareConfig:
    return CompareConfig(groups=("city",), aggregates=("price_issue", "quality_issue"))


@pytest.fixture()
def temporal_config() -> CompareConfig:
    return CompareConfig(
        groups=("city",), aggregates=("price_issue", "quality_issue"), temporal="period"
    )


@pytest.fixture()
def fake() -> FakeChat:
    return FakeChat()


@pytest.fixture()
def result(complaints: pl.DataFrame, config: CompareConfig) -> CompareResult:
    return CaseGroupComparer.summarize(complaints, config)


@pytest.fixture()
def temporal_result(
    complaints: pl.DataFrame, temporal_config: CompareConfig
) -> CompareResult:
    return CaseGroupComparer.summarize(complaints, temporal_config)
