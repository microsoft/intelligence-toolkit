# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from unittest.mock import patch

import pandas as pd
import pytest

from intelligence_toolkit.detect_case_patterns.detection_functions import (
    create_close_node_rows,
    create_pattern_rows,
    create_period_to_patterns,
)

SEP = "=="


class FakeRecordCounter:
    """Minimal stand-in for RecordCounter with fixed counts and stats."""

    def __init__(self, counts=None, default_count=0, stats=(1.0, 1.0, 1.0)):
        self.counts = counts or {}
        self.default_count = default_count
        self.stats = stats

    def count_records(self, atts):
        return self.counts.get(tuple(atts), self.default_count)

    def compute_period_mean_sd_max(self, atts):
        return self.stats


@pytest.fixture()
def always_converging():
    with patch(
        "intelligence_toolkit.detect_case_patterns.detection_functions.is_converging_pair",
        return_value=True,
    ) as mock:
        yield mock


def test_create_close_node_rows_skips_same_attribute_pairs(always_converging):
    nodes = [f"color{SEP}red", f"color{SEP}blue", f"size{SEP}big"]
    rc = FakeRecordCounter(default_count=5, stats=(2.0, 1.0, 5.0))

    df, all_pairs, close_pairs = create_close_node_rows(
        ["2020"], {}, nodes, min_pattern_count=1, rc=rc, type_val_sep=SEP
    )

    assert all_pairs == 3
    # color==red/color==blue share an attribute type and are never compared
    assert close_pairs == 2
    assert set(zip(df["node1"], df["node2"], strict=True)) == {
        (f"color{SEP}red", f"size{SEP}big"),
        (f"color{SEP}blue", f"size{SEP}big"),
    }


def test_create_close_node_rows_computes_count_metrics(always_converging):
    nodes = [f"a{SEP}1", f"b{SEP}1"]
    rc = FakeRecordCounter(default_count=6, stats=(2.0, 1.0, 6.0))

    df, _, _ = create_close_node_rows(
        ["2020"], {}, nodes, min_pattern_count=1, rc=rc, type_val_sep=SEP
    )

    row = df.iloc[0]
    assert row["period_count"] == 6
    assert row["count_delta"] == 4.0
    assert row["count_factor"] == 3.0


def test_create_close_node_rows_respects_min_pattern_count(always_converging):
    nodes = [f"a{SEP}1", f"b{SEP}1"]
    rc = FakeRecordCounter(default_count=2)

    df, all_pairs, close_pairs = create_close_node_rows(
        ["2020"], {}, nodes, min_pattern_count=3, rc=rc, type_val_sep=SEP
    )

    assert all_pairs == 1
    assert close_pairs == 0
    assert df.empty


def test_create_close_node_rows_ignores_non_converging_pairs():
    nodes = [f"a{SEP}1", f"b{SEP}1"]
    with patch(
        "intelligence_toolkit.detect_case_patterns.detection_functions.is_converging_pair",
        return_value=False,
    ):
        df, _, close_pairs = create_close_node_rows(
            ["2020"], {}, nodes, min_pattern_count=1, rc=FakeRecordCounter(default_count=9), type_val_sep=SEP
        )

    assert close_pairs == 0
    assert df.empty


def _close_node_df(period, pairs):
    return pd.DataFrame(
        [[period, a, b, 10] for a, b in pairs],
        columns=["period", "node1", "node2", "period_count"],
    )


def test_create_period_to_patterns_builds_fully_connected_triple():
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    rc = FakeRecordCounter(default_count=10)

    result = create_period_to_patterns(
        ["2020"], _close_node_df("2020", pairs), max_pattern_length=3, min_pattern_count=1, rc=rc
    )

    patterns = [p for p, _ in result["2020"]]
    assert ["A", "B", "C"] in patterns
    assert ["A", "B"] in patterns


def test_create_period_to_patterns_rejects_incomplete_cliques():
    # A-B and B-C are close but A-C is not, so A,B,C must not form a pattern
    pairs = [("A", "B"), ("B", "C")]
    rc = FakeRecordCounter(default_count=10)

    result = create_period_to_patterns(
        ["2020"], _close_node_df("2020", pairs), max_pattern_length=3, min_pattern_count=1, rc=rc
    )

    assert ["A", "B", "C"] not in [p for p, _ in result["2020"]]


def test_create_period_to_patterns_respects_max_length():
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    rc = FakeRecordCounter(default_count=10)

    result = create_period_to_patterns(
        ["2020"], _close_node_df("2020", pairs), max_pattern_length=2, min_pattern_count=1, rc=rc
    )

    assert all(len(p) <= 2 for p, _ in result["2020"])


def test_create_pattern_rows_scores_and_filters():
    period_to_patterns = {
        "2020": [
            ([], 0),  # seed entry, skipped
            (["A", "B"], 7),  # score (7-3)/2 = 2.0, kept
            (["A", "C"], 1),  # score (1-3)/2 = -1.0, dropped
        ]
    }
    rc = FakeRecordCounter(stats=(3.0, 2.0, 7.0))

    rows = create_pattern_rows(period_to_patterns, rc)

    assert rows == [["2020", "A & B", 2, 7, 3.0, 2.0]]
