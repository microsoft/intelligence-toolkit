# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Fixtures: a small categorical frame and a fake synthesizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from itk_anonymize import AnonymizeConfig, Anonymizer, AnonymizeResult
from itk_anonymize_testing import FakeSynthesizer, complaints_frame

if TYPE_CHECKING:
    import polars as pl


@pytest.fixture()
def complaints() -> pl.DataFrame:
    return complaints_frame()


@pytest.fixture()
def fake() -> FakeSynthesizer:
    return FakeSynthesizer()


@pytest.fixture()
def config() -> AnonymizeConfig:
    return AnonymizeConfig(epsilon=4.0, reporting_length=3)


@pytest.fixture()
def result(
    complaints: pl.DataFrame, fake: FakeSynthesizer, config: AnonymizeConfig
) -> AnonymizeResult:
    return Anonymizer(fake).anonymize(complaints, config)
