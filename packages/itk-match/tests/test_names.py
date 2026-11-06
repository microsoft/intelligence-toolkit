# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pytest
from itk_match.names import (
    SCORERS,
    fuzzy_ratio,
    get_scorer,
    score_pairs,
    trigram_jaccard,
)


def test_trigram_jaccard_matches_documented_examples():
    # Worked examples from the Match Entity Records tutorial.
    assert trigram_jaccard("Sprout Ltd", "Spout Ltd") == pytest.approx(0.5)
    assert trigram_jaccard(
        "Sprout Technology Limited", "Spout Technology Limited"
    ) == pytest.approx(0.8)


def test_trigram_jaccard_ignores_case_and_punctuation():
    assert trigram_jaccard("ACME, Inc.", "acme inc") == 1.0


def test_trigram_jaccard_handles_short_and_empty_names():
    assert trigram_jaccard("ab", "cd") == 0.0
    assert trigram_jaccard("", "") == 1.0


def test_fuzzy_ratio_is_word_order_tolerant():
    assert fuzzy_ratio("Gear Adventure", "Adventure Gear") == 1.0
    assert 0.0 <= fuzzy_ratio("Alpha", "Zeta") < 0.5


def test_get_scorer_exposes_both_scorers():
    assert get_scorer("trigram") is trigram_jaccard
    assert get_scorer("fuzzy") is fuzzy_ratio
    assert set(SCORERS) == {"trigram", "fuzzy"}


def test_score_pairs_attaches_name_similarity():
    names = ["Sprout Ltd", "Spout Ltd", "Other"]
    pairs = score_pairs([(0, 1, 0.01), (0, 2, 0.02)], names, trigram_jaccard)
    assert [(p.left, p.right) for p in pairs] == [(0, 1), (0, 2)]
    assert pairs[0].name_similarity == pytest.approx(0.5)
    assert pairs[0].distance == 0.01
