# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Lexical similarity between entity names."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from itk_match.models import Pair

if TYPE_CHECKING:
    from itk_match.blocking import Candidate
    from itk_match.models import NameScorerName

_PUNCTUATION = re.compile(r"[^\w\s]")


class NameScorer(Protocol):
    def __call__(self, left: str, right: str) -> float: ...


def _clean(name: str) -> str:
    return _PUNCTUATION.sub("", name.upper()).strip()


def trigram_jaccard(left: str, right: str, n: int = 3) -> float:
    """Jaccard similarity of character n-grams; identical names score 1.0."""
    a, b = _clean(left), _clean(right)
    if a == b:
        return 1.0
    grams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
    union = grams_a | grams_b
    return len(grams_a & grams_b) / len(union) if union else 0.0


def fuzzy_ratio(left: str, right: str) -> float:
    """rapidfuzz token-set ratio scaled to 0..1, tolerant of word order."""
    from rapidfuzz import fuzz

    return fuzz.token_set_ratio(_clean(left), _clean(right)) / 100.0


SCORERS: dict[str, NameScorer] = {"trigram": trigram_jaccard, "fuzzy": fuzzy_ratio}


def get_scorer(name: NameScorerName) -> NameScorer:
    return SCORERS[name]


def score_pairs(
    candidates: list[Candidate], names: list[str], scorer: NameScorer
) -> list[Pair]:
    """Attach a name similarity to each candidate pair."""
    return [
        Pair(left, right, distance, scorer(names[left], names[right]))
        for left, right, distance in candidates
    ]
