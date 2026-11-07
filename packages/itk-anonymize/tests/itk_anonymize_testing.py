# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Test helpers shared by the itk-anonymize suite.

Lives outside conftest.py (and the tests directory has no __init__.py) so it
cannot collide with the ``tests`` packages of sibling workspace members.
"""

from __future__ import annotations

import random

import polars as pl
from itk_anonymize.aggregate import count_combinations, render_key
from itk_anonymize.models import AnonymizeConfig, RawTable, SynthesisOutput


class FakeSynthesizer:
    """Deterministic stand-in for pacsynth; never runs differential privacy.

    ``dp_aggregates`` are the plain counts of the input. With ``noise`` every
    third combination (in sorted key order) is bumped by one, every third
    other one is dropped (suppressed), and one fabricated combination is
    added. The synthetic table is the input shuffled by ``seed`` with the last
    row dropped when ``drop_last``.
    """

    def __init__(
        self, noise: bool = False, seed: int = 0, drop_last: bool = False
    ) -> None:
        self.noise = noise
        self.seed = seed
        self.drop_last = drop_last
        self.calls: list[tuple[RawTable, AnonymizeConfig]] = []

    def synthesize(self, table: RawTable, config: AnonymizeConfig) -> SynthesisOutput:
        self.calls.append((table, config))
        plain = count_combinations(
            table,
            config.reporting_length,
            config.value_separator,
            config.sensitive_zeros,
        )
        raw: dict[str, int] = {}
        for index, key in enumerate(sorted(plain)):
            count = plain[key]
            if self.noise and index % 3 == 1:
                count += 1
            if self.noise and index % 3 == 2:
                continue
            raw[render_key(key, config.attribute_separator)] = count
        if self.noise:
            raw[f"fabricated{config.value_separator}yes"] = 2
        rows = list(table.rows)
        random.Random(self.seed).shuffle(rows)
        if self.drop_last and rows:
            rows = rows[:-1]
        return SynthesisOutput(raw, table.height, RawTable(table.header, tuple(rows)))


def complaints_frame() -> pl.DataFrame:
    """Eight already-categorical customer complaint records."""
    return pl.DataFrame({
        "city": [
            "Lakeside",
            "Lakeside",
            "Hilltop",
            "Hilltop",
            "Lakeside",
            "Seaside",
            "Hilltop",
            "Seaside",
        ],
        "product": ["A", "B", "A", "A", "B", "A", "B", ""],
        "price_issue": ["True", "", "True", "", "True", "", "", "True"],
        "period": [
            "2023-H1",
            "2023-H1",
            "2023-H2",
            "2023-H2",
            "2023-H1",
            "2023-H2",
            "2023-H1",
            "2023-H2",
        ],
    })


def av(attribute: str, value: str):
    from itk_anonymize.models import AttributeValue

    return AttributeValue(attribute=attribute, value=value)
