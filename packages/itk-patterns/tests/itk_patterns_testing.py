# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Test helpers shared by the itk-patterns suite.

Lives outside conftest.py (and the tests directory has no __init__.py) so it
cannot collide with the ``tests`` packages of sibling workspace members.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx
    from itk_patterns.models import PatternConfig

PERIODS = ("P1", "P2", "P3")


def cases_frame() -> pl.DataFrame:
    """Thirteen cases over three periods; ``city=A & product=X`` spikes in P3.

    Per-period counts: ``city=A & product=X`` 1, 1, 4; ``city=A & issue=high``
    1, 0, 3; ``product=X & issue=high`` 0, 1, 3; all three together 0, 0, 3.
    """
    rows = [
        ("P1", "A", "X", "low"),
        ("P1", "B", "Y", "low"),
        ("P1", "A", "Y", "high"),
        ("P1", "B", "X", "low"),
        ("P2", "A", "X", "low"),
        ("P2", "B", "Y", "high"),
        ("P2", "A", "Y", "low"),
        ("P2", "B", "X", "high"),
        ("P3", "A", "X", "high"),
        ("P3", "A", "X", "high"),
        ("P3", "A", "X", "low"),
        ("P3", "B", "Y", "low"),
        ("P3", "A", "X", "high"),
    ]
    return pl.DataFrame({
        "period": [r[0] for r in rows],
        "city": [r[1] for r in rows],
        "product": [r[2] for r in rows],
        "issue": [r[3] for r in rows],
    })


# Position per node and period; every other node sits at DEFAULT_POSITION.
# Cosine distance between the second components alone depends only on sign,
# so the three nodes with a positive second component converge in P3 while
# their all-time centroids differ. ``(3, -4)`` has norm 5, so nodes that never
# move share a centroid whose self-distance is exactly 0, not 2e-16.
DEFAULT_POSITION = (3.0, -4.0)
HOT_POSITIONS: dict[str, dict[str, tuple[float, float]]] = {
    "P3": {"city=A": (3.0, 4.0), "product=X": (0.0, 4.0), "issue=high": (4.0, 3.0)}
}


class FakeEmbedder:
    """Deterministic :class:`itk_patterns.embed.PeriodEmbedder`.

    Every node's row is ``positions[period][node]`` (the period read from
    ``graph.graph``), defaulting to ``DEFAULT_POSITION``. Records the period
    of every call.
    """

    def __init__(
        self, positions: dict[str, dict[str, tuple[float, float]]] | None = None
    ) -> None:
        self.positions = HOT_POSITIONS if positions is None else positions
        self.calls: list[str] = []

    def embed(
        self,
        graph: nx.Graph,
        nodes: Sequence[str],
        labels: Sequence[int],
        config: PatternConfig,
    ) -> np.ndarray:
        period = str(graph.graph.get("period", ""))
        self.calls.append(period)
        hot = self.positions.get(period, {})
        return np.array([hot.get(node, DEFAULT_POSITION) for node in nodes])


class FakeChat:
    """Records the messages it was asked to answer and returns a fixed report."""

    def __init__(self, answer: str = "# Pattern Report") -> None:
        self.answer = answer
        self.messages: list[list[dict[str, str]]] = []

    async def __call__(self, messages, stream=False, callbacks=None) -> str:
        self.messages.append(messages)
        return self.answer
