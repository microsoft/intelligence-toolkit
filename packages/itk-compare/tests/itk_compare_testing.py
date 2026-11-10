# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Test helpers shared by the itk-compare suite.

Lives outside conftest.py (and the tests directory has no __init__.py) so it
cannot collide with the ``tests`` packages of sibling workspace members.
"""

from __future__ import annotations

import polars as pl


class FakeChat:
    """Records the messages it was asked and answers with a fixed report."""

    def __init__(self, answer: str = "# Report\n\nGroups compared.") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages, callbacks=None) -> str:
        self.calls.append(messages)
        for callback in callbacks or []:
            callback.on_llm_new_token(self.answer)
        return self.answer


def complaints_frame() -> pl.DataFrame:
    """Eight complaint records; the last has no city and must be dropped."""
    return pl.DataFrame({
        "city": [
            "Lakeside",
            "Lakeside",
            "Lakeside",
            "Hilltop",
            "Hilltop",
            "Hilltop",
            "Seaside",
            "",
        ],
        "product": ["A", "B", "A", "A", "B", "A", "A", "B"],
        "price_issue": ["True", "True", "False", "True", "", "True", "False", "True"],
        "quality_issue": ["False", "", "True", "True", "True", "True", "False", "True"],
        "period": [
            "2023-H1",
            "2023-H1",
            "2023-H2",
            "2023-H1",
            "2023-H2",
            "2023-H2",
            "2023-H2",
            "2023-H1",
        ],
    })


def rows(frame: pl.DataFrame, *columns: str) -> list[tuple]:
    """Row tuples of the given columns, for compact assertions."""
    return frame.select(list(columns)).rows()
