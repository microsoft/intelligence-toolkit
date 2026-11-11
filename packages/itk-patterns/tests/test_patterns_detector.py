# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio

import pytest
from itk_patterns import DetectionConfig, PatternDetector
from itk_patterns.embed import EncoderEmbedder
from itk_patterns.models import PATTERN, PERIOD, PatternResult
from itk_patterns_testing import FakeChat, FakeEmbedder


def test_prepare_builds_links_graphs_and_labels(prepared, cases):
    assert isinstance(prepared, PatternResult)
    assert prepared.source.height == cases.height
    assert prepared.periods == ["P1", "P2", "P3"]
    assert len(prepared.attributes) == 6
    assert prepared.positions is None
    summary = prepared.summary
    assert (summary.links, summary.cases, summary.attributes, summary.periods) == (
        39,
        13,
        6,
        3,
    )
    assert str(summary).startswith("Attribute model has 39 links")


def test_embed_stores_positions_and_clears_detections(prepared, fake):
    embedded = PatternDetector(fake).embed(prepared)
    assert embedded.positions is not None
    assert embedded.shifts is not None
    assert set(embedded.positions) == set(prepared.attributes)
    assert embedded.patterns.height == 0
    assert embedded.detection is None
    assert fake.calls == ["P1", "P2", "P3"]
    # the input result is untouched
    assert prepared.positions is None


def test_detect_requires_embedding(prepared):
    with pytest.raises(ValueError, match="not been embedded"):
        PatternDetector.detect(prepared)


def test_detect_fills_patterns_pairs_and_series(detected, detection):
    assert detected.detection == detection
    assert detected.close_pairs == 6
    assert detected.all_pairs == 45
    assert detected.patterns.get_column(PATTERN).to_list()[0] == (
        "city=A & issue=high & product=X"
    )
    assert detected.patterns.get_column(PERIOD).unique().to_list() == ["P3"]
    # one series per distinct pattern, three periods each
    assert detected.time_series.height == 4 * 3
    series = detected.pattern_series("city=A & product=X")
    assert series.get_column("count").to_list() == [1, 1, 4]
    shown = detected.patterns_display
    assert shown.get_column("overall_score").to_list()[0] == 1.0


def test_detect_reruns_with_new_thresholds_without_re_embedding(embedded, fake):
    calls = len(fake.calls)
    loose = PatternDetector.detect(embedded, DetectionConfig(min_pattern_count=2))
    strict = PatternDetector.detect(
        embedded, DetectionConfig(min_pattern_count=3, max_pattern_length=2)
    )
    assert len(fake.calls) == calls
    assert loose.patterns.height == 4
    assert strict.patterns.get_column(PATTERN).to_list() == ["city=A & product=X"]
    assert embedded.patterns.height == 0


def test_detect_defaults_to_detection_config_defaults(embedded):
    detected = PatternDetector.detect(embedded)
    assert detected.detection == DetectionConfig()
    assert detected.patterns.height == 0
    assert detected.time_series.height == 0


def test_default_embedder_is_the_encoder_and_runs_without_network(prepared):
    detector = PatternDetector()
    assert isinstance(detector.embedder, EncoderEmbedder)
    embedded = detector.embed(prepared)
    assert embedded.positions is not None
    assert embedded.positions["city=A"]["P1"].shape == (3,)
    detected = PatternDetector.detect(embedded, DetectionConfig(min_pattern_count=1))
    assert detected.all_pairs == 45


def test_attribute_counts_and_report_messages(detected):
    counts = PatternDetector.attribute_counts(detected, "city=A & product=X", "P3")
    assert counts.row(0) == ("city=A", 4)
    messages = PatternDetector.report_messages(detected, "city=A & product=X", "P3")
    content = messages[0]["content"]
    assert "Detected pattern: city=A & product=X" in content
    assert "P3,city=A & product=X,4" in content
    assert "issue=high,3" in content


def test_report_uses_chat_and_requires_one(detected):
    chat = FakeChat("report text")
    detector = PatternDetector(FakeEmbedder(), chat)
    answer = asyncio.run(detector.report(detected, "city=A & product=X", "P3"))
    assert answer == "report text"
    assert "Detected in period: P3" in chat.messages[0][0]["content"]
    with pytest.raises(ValueError, match="no chat function"):
        asyncio.run(PatternDetector().report(detected, "city=A & product=X", "P3"))
