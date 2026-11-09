# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Test helpers shared by the itk-networks suite.

Lives outside conftest.py (and the tests directory has no __init__.py) so it
cannot collide with the ``tests`` packages of sibling workspace members.
"""

from __future__ import annotations

import string
from typing import Any

import networkx as nx
import numpy as np
import polars as pl
from itk_networks.models import AttributeTable, FlagFormat, FlagTable, GroupTable
from itk_networks.text import qualify, qualify_entity

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
    """Stands in for BaseEmbedder.embed_store_many without disk or network."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[str]] = []

    async def embed_store_many(
        self, data: list[Any], callbacks: Any = None, cache_data: bool = True
    ) -> list[dict[str, Any]]:
        self.calls.append([item["text"] for item in data])
        return [
            {
                "text": item["text"],
                "hash": item["hash"],
                "vector": self.vectors.get(item["text"]) or letter_bag(item["text"]),
            }
            for item in data
        ]


class ChunkDetector:
    """Deterministic CommunityDetector: sorted nodes sliced into blocks of max_size."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def detect(self, graph: nx.Graph, max_size: int, seed: int) -> dict[str, int]:
        self.calls.append((graph.number_of_nodes(), max_size, seed))
        return {node: ix // max_size for ix, node in enumerate(sorted(graph.nodes()))}


class FakeChat:
    """Returns a canned report and records what it was asked."""

    def __init__(self, reply: str = "##### Evaluation of Entity Network 0") -> None:
        self.reply = reply
        self.messages: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def __call__(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        callbacks: Any = None,
    ) -> str:
        self.messages.append(messages)
        self.kwargs.append({"stream": stream, "callbacks": callbacks})
        return self.reply


def E(entity: str) -> str:  # noqa: N802 - reads like the node it builds
    return qualify_entity(entity)


def A(attribute: str, value: str) -> str:  # noqa: N802
    return qualify(attribute, value)


def sample_frame() -> pl.DataFrame:
    """Six companies: A/B share a phone, C/D share phone and email, E/F are alone.

    Every row shares ``city`` so it is trimmed at ``max_attribute_degree=3``.
    """
    return pl.DataFrame({
        "name": ["A", "B", "C", "D", "E", "F"],
        "phone": ["111", "111", "222", "222", "333", "444"],
        "email": ["a@x", "b@x", "c@x", "c@x", "e@x", None],
        "city": ["Metro"] * 6,
        "safety": [2, 0, 1, 0, 0, 0],
        "pay": [0, 0, 0, 0, 0, 1],
        "sector": ["retail", "retail", "mining", "mining", "retail", None],
    })


def attribute_table(frame: pl.DataFrame | None = None) -> AttributeTable:
    return AttributeTable(
        frame=frame if frame is not None else sample_frame(),
        entity_column="name",
        columns=["phone", "email", "city"],
    )


def flag_table(
    frame: pl.DataFrame | None = None, fmt: FlagFormat = FlagFormat.Count
) -> FlagTable:
    return FlagTable(
        frame=frame if frame is not None else sample_frame(),
        entity_column="name",
        columns=["safety", "pay"],
        format=fmt,
    )


def group_table(frame: pl.DataFrame | None = None) -> GroupTable:
    return GroupTable(
        frame=frame if frame is not None else sample_frame(),
        entity_column="name",
        columns=["sector"],
    )


def path_graph(*nodes: str) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node, type=node.split("==")[0])
    nx.add_path(graph, nodes)
    return graph
