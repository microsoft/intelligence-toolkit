# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Shared fixtures: a deterministic embedder and two overlapping datasets."""

from __future__ import annotations

import string
from typing import Any

import numpy as np
import polars as pl
import pytest
from itk_match.models import Dataset

_ALPHABET = string.ascii_uppercase + string.digits


def letter_bag(text: str) -> list[float]:
    """Character-frequency vector: near-identical strings land close together."""
    vector = np.zeros(len(_ALPHABET), dtype=np.float32)
    for ch in text.upper():
        idx = _ALPHABET.find(ch)
        if idx >= 0:
            vector[idx] += 1.0
    return vector.tolist()


class FakeEmbedder:
    """Stands in for BaseEmbedder.embed_store_many without touching disk or APIs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_store_many(
        self, data: list[Any], callbacks: Any = None, cache_data: bool = True
    ) -> list[dict[str, Any]]:
        self.calls.append([item["text"] for item in data])
        return [
            {
                "text": item["text"],
                "hash": item["hash"],
                "vector": letter_bag(item["text"]),
            }
            for item in data
        ]


class FakeChat:
    """Returns a canned JSON evaluation for every group_id it is shown."""

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.messages: list[list[dict[str, str]]] = []

    async def __call__(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        callbacks: Any = None,
    ) -> str:
        self.messages.append(messages)
        if self.reply is not None:
            return self.reply
        csv_text = messages[-1]["content"]
        ids = sorted({
            int(line.split(",")[0]) for line in csv_text.splitlines()[3:] if line
        })
        rows = ",".join(
            f'{{"group_id": {i}, "relatedness": 8, "explanation": "same address"}}'
            for i in ids
        )
        return f'```json\n{{"evaluations": [{rows}]}}\n```'


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def dataset_one() -> Dataset:
    frame = pl.DataFrame({
        "employer_id": [1, 2, 3],
        "employer_name": ["Adventure Gear", "Sprout Ltd", "MediaWave"],
        "address": ["1010 Trekking Trl", "5 Seed St", "1111 Broadcast Blvd"],
        "city": ["Adventure City", "Greentown", "Media City"],
    })
    return Dataset(
        name="D1",
        frame=frame,
        name_column="employer_name",
        id_column="employer_id",
        attributes={"address": "address", "city": "city"},
    )


@pytest.fixture()
def dataset_two() -> Dataset:
    frame = pl.DataFrame({
        "company_name": ["Adventure Gear", "Spout Ltd", "UrbanTech", ""],
        "street_address": ["1010 Trekking Trl", "5 Seed St", "909 Innovation Dr", "x"],
        "city_address": ["Adventure City", "Greentown", "Tech City", "y"],
        "phone_number": [111, 222, 333, 444],
    })
    return Dataset(
        name="D2",
        frame=frame,
        name_column="company_name",
        attributes={
            "street_address": "address",
            "city_address": "city",
            "phone_number": "phone",
        },
    )
