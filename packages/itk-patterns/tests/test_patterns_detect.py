# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import math

import numpy as np
import polars as pl
import pytest
from itk_patterns import DetectionConfig
from itk_patterns.detect import (
    RecordCounter,
    converging_pairs,
    detect_patterns,
    display_patterns,
    grow_patterns,
    node_types,
    score_patterns,
)
from itk_patterns.models import (
    COUNT,
    DETECTIONS,
    LENGTH,
    MEAN,
    NODE1,
    NODE2,
    OVERALL_SCORE,
    PAIR_SCHEMA,
    PATTERN,
    PATTERN_SCHEMA,
    PERIOD,
    PERIOD_COUNT,
    Z_SCORE,
)


@pytest.fixture()
def counter(prepared) -> RecordCounter:
    return RecordCounter(prepared.links)


def test_record_counter_intersects_attributes_and_periods(counter):
    assert counter.periods == ["P1", "P2", "P3"]
    assert counter.count_records(["city=A"]) == 8
    assert counter.count_records(["city=A", "product=X"]) == 6
    assert counter.count_records(["P3", "city=A", "product=X"]) == 4
    assert counter.count_records(["P2", "city=A", "issue=high"]) == 0
    assert counter.count_records(["nope"]) == 0
    assert counter.count_records([]) == 0


def test_record_counter_period_statistics(counter):
    assert counter.period_counts(["city=A", "product=X"]) == [1, 1, 4]
    mean, sd, top = counter.period_mean_sd_max(["city=A", "product=X"])
    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx(math.sqrt(2.0))
    assert top == 4


def test_record_counter_caches_by_term_set(counter):
    first = counter.count_records(["product=X", "city=A"])
    assert counter.count_records(["city=A", "product=X"]) == first
    assert len(counter._cache) == 1  # noqa: SLF001


def test_node_types_maps_each_attribute(prepared):
    assert node_types(prepared.links)["issue=high"] == "issue"


def test_converging_pairs_filters_type_sign_and_count(embedded, config, detection):
    counter = RecordCounter(embedded.links)
    assert embedded.positions is not None
    pairs, all_pairs = converging_pairs(
        embedded.positions, embedded.links, counter, config, detection
    )
    assert pairs.schema == PAIR_SCHEMA
    assert all_pairs == 3 * math.comb(6, 2)
    found = set(pairs.select([PERIOD, NODE1, NODE2]).rows())
    # P1 and P2: every node is negative, so pairs with one moving node converge
    # and only those two subjects or more carry survive; P3: the three positive nodes.
    assert found == {
        ("P1", "issue=low", "product=X"),
        ("P2", "city=A", "issue=low"),
        ("P2", "city=B", "issue=high"),
        ("P3", "city=A", "issue=high"),
        ("P3", "city=A", "product=X"),
        ("P3", "issue=high", "product=X"),
    }
    row = pairs.filter(pl.col(NODE2) == "product=X").filter(pl.col(NODE1) == "city=A")
    assert row.get_column(PERIOD_COUNT).item() == 4
    assert row.get_column("mean_count").item() == pytest.approx(2.0)
    assert row.get_column("count_delta").item() == pytest.approx(2.0)
    assert row.get_column("count_factor").item() == pytest.approx(2.0)


def test_converging_pairs_factor_is_null_when_mean_is_zero(config, detection):
    links = pl.DataFrame({
        "subject_id": [1, 1],
        "period": ["P1", "P1"],
        "attribute_type": ["a", "b"],
        "attribute_value": ["x", "y"],
        "full_attribute": ["a=x", "b=y"],
    })
    positions = {
        "a=x": {"P1": np.array([1.0, 1.0]), "ALL": np.array([1.0, 1.0])},
        "b=y": {"P1": np.array([1.0, 1.0]), "ALL": np.array([1.0, 0.5])},
    }
    pairs, _ = converging_pairs(
        positions,
        links,
        RecordCounter(links),
        config,
        DetectionConfig(min_pattern_count=1),
    )
    assert pairs.height == 1
    assert pairs.get_column("count_factor").item() == pytest.approx(1.0)


def test_grow_patterns_extends_pairs_into_triples(embedded, detection):
    counter = RecordCounter(embedded.links)
    assert embedded.positions is not None
    pairs, _ = converging_pairs(
        embedded.positions, embedded.links, counter, embedded.config, detection
    )
    found = grow_patterns(pairs, counter, detection)
    # the P1 and P2 pairs have exactly min_pattern_count subjects: not strictly more
    assert found["P1"] == []
    assert found["P2"] == []
    assert found["P3"] == [
        (("city=A", "issue=high"), 3),
        (("city=A", "product=X"), 4),
        (("issue=high", "product=X"), 3),
        (("city=A", "issue=high", "product=X"), 3),
    ]


def test_grow_patterns_respects_max_length_and_strict_min_count(embedded):
    counter = RecordCounter(embedded.links)
    assert embedded.positions is not None
    pairs, _ = converging_pairs(
        embedded.positions,
        embedded.links,
        counter,
        embedded.config,
        DetectionConfig(min_pattern_count=2, max_pattern_length=2),
    )
    short = grow_patterns(
        pairs, counter, DetectionConfig(min_pattern_count=2, max_pattern_length=2)
    )
    assert all(len(p) == 2 for p, _ in short["P3"])
    # count 3 is not strictly greater than 3, so the triples and two pairs drop out
    strict = grow_patterns(
        pairs, counter, DetectionConfig(min_pattern_count=3, max_pattern_length=3)
    )
    assert strict["P3"] == [(("city=A", "product=X"), 4)]


def test_score_patterns_z_scores_detections_and_normalised_overall(counter):
    found = {
        "P3": [
            (("city=A", "product=X"), 4),
            (("city=A", "issue=high", "product=X"), 3),
        ]
    }
    patterns = score_patterns(found, counter)
    assert patterns.schema == PATTERN_SCHEMA
    by_name = {row[PATTERN]: row for row in patterns.to_dicts()}
    pair = by_name["city=A & product=X"]
    assert pair[COUNT] == 4
    assert pair[LENGTH] == 2
    assert pair[DETECTIONS] == 1
    assert pair[MEAN] == pytest.approx(2.0)
    assert pair[Z_SCORE] == pytest.approx(math.sqrt(2.0))
    triple = by_name["city=A & issue=high & product=X"]
    assert triple[Z_SCORE] == pytest.approx(math.sqrt(2.0))
    raw_pair = math.sqrt(2) * 2 * 1 * math.log1p(4)
    raw_triple = math.sqrt(2) * 3 * 1 * math.log1p(3)
    assert triple[OVERALL_SCORE] == pytest.approx(1.0)
    assert pair[OVERALL_SCORE] == pytest.approx(raw_pair / raw_triple)
    assert (
        patterns.get_column(PATTERN).to_list()[0] == "city=A & issue=high & product=X"
    )


def test_score_patterns_counts_detections_across_periods(counter):
    found = {
        "P1": [(("city=A", "product=X"), 1)],
        "P3": [(("city=A", "product=X"), 4)],
    }
    patterns = score_patterns(found, counter)
    # P1's count of 1 is below the mean, so only P3 survives, yet both count
    # as detections only when both rows survive
    assert patterns.get_column(DETECTIONS).to_list() == [1]


def test_score_patterns_drops_constant_and_negative(counter):
    found = {"P1": [(("city=B", "product=Y"), 1)], "P2": [(("city=A", "product=X"), 1)]}
    # city=B & product=Y occurs once in every period: sd 0; A&X in P2 is below its mean
    assert score_patterns(found, counter).height == 0
    assert score_patterns({}, counter).schema == PATTERN_SCHEMA


def test_display_patterns_rounds_like_legacy():
    frame = pl.DataFrame(
        {
            PERIOD: ["P"],
            PATTERN: ["a=1 & b=2"],
            LENGTH: [2],
            COUNT: [4],
            MEAN: [2.4],
            Z_SCORE: [1.23456],
            DETECTIONS: [1],
            OVERALL_SCORE: [0.98765],
        },
        schema=PATTERN_SCHEMA,
    )
    shown = display_patterns(frame).row(0, named=True)
    assert (shown[MEAN], shown[Z_SCORE], shown[OVERALL_SCORE]) == (2.0, 1.23, 0.99)


def test_detect_patterns_end_to_end(embedded, detection):
    assert embedded.positions is not None
    patterns, pairs, all_pairs = detect_patterns(
        embedded.positions, embedded.links, embedded.config, detection
    )
    assert pairs.height == 6
    assert all_pairs == 45
    assert patterns.get_column(PATTERN).to_list() == [
        "city=A & issue=high & product=X",
        "city=A & product=X",
        "city=A & issue=high",
        "issue=high & product=X",
    ]
