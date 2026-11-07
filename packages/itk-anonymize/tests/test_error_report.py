# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import pytest
from itk_anonymize.evaluate import error_report


def test_per_length_rows_then_overall() -> None:
    source = {("A", "B"): 10, ("C",): 5, ("D", "E"): 20, ("F",): 15}
    target = {("A", "B"): 12, ("C",): 5, ("D", "E"): 20, ("G",): 3}
    report = error_report(source, target)

    lengths = [row.length for row in report.rows]
    assert lengths == [1, 2, None]

    one = report.rows[0]
    assert one.mean_count == 10.0  # (5 + 15) / 2
    assert one.mean_error == 0.0  # only C overlaps, exactly
    assert one.suppressed_pct == pytest.approx(15 * 100 / 20)  # F
    assert one.fabricated_pct == pytest.approx(3 * 100 / 8)  # G out of C + G

    two = report.rows[1]
    assert two.mean_count == 15.0
    assert two.mean_error == 1.0  # (2 + 0) / 2
    assert two.suppressed_pct == 0.0
    assert two.fabricated_pct == 0.0

    overall = report.overall
    assert overall.mean_count == 12.5
    assert overall.mean_error == pytest.approx(2 / 3)
    assert overall.suppressed_pct == pytest.approx(15 * 100 / 50)
    assert overall.fabricated_pct == pytest.approx(3 * 100 / 40)


def test_lengths_without_overlap_get_no_row() -> None:
    source = {("A",): 1, ("B", "C"): 1, ("X", "Y", "Z"): 4}
    target = {("A",): 1}
    report = error_report(source, target)
    assert [row.length for row in report.rows] == [1, None]
    # ...but they still count towards the overall suppressed figure
    assert report.overall.suppressed_pct == pytest.approx(5 * 100 / 6)


def test_no_overlap_gives_null_error_not_nan() -> None:
    report = error_report({("A",): 1}, {("B",): 1})
    assert report.rows == (report.overall,)
    assert report.overall.mean_error is None
    assert report.overall.suppressed_pct == 100.0
    assert report.overall.fabricated_pct == 100.0


def test_empty_inputs_give_nulls_not_exceptions() -> None:
    report = error_report({}, {})
    assert report.overall.mean_count is None
    assert report.overall.suppressed_pct is None
    assert report.overall.fabricated_pct is None


def test_text_frame_matches_legacy_strings() -> None:
    report = error_report({("A",): 10, ("B", "C"): 4}, {("A",): 12, ("B", "C"): 4})
    text = report.to_text_frame()
    assert text.columns == ["Length", "Count +/- Error", "Suppressed %", "Fabricated %"]
    assert text.rows() == [
        ("1", "10.00 +/- 2.00", "0.00 %", "0.00 %"),
        ("2", "4.00 +/- 0.00", "0.00 %", "0.00 %"),
        ("Overall", "7.00 +/- 1.00", "0.00 %", "0.00 %"),
    ]


def test_text_frame_renders_missing_as_nan_like_legacy() -> None:
    text = error_report({("A",): 1}, {("B",): 1}).to_text_frame()
    assert text.row(0) == ("Overall", "1.00 +/- nan", "100.00 %", "100.00 %")


def test_numeric_frame() -> None:
    frame = error_report({("A",): 10}, {("A",): 12}).frame
    assert frame.columns == [
        "length",
        "mean_count",
        "mean_error",
        "suppressed_pct",
        "fabricated_pct",
    ]
    assert frame.rows() == [(1, 10.0, 2.0, 0.0, 0.0), (None, 10.0, 2.0, 0.0, 0.0)]
