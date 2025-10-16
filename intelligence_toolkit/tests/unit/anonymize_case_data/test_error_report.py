# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
import pandas as pd
import numpy as np
from collections import defaultdict
from intelligence_toolkit.anonymize_case_data.error_report import ErrorReport


def test_error_report_initialization():
    src_aggs = {("A", "B"): 10, ("C",): 5}
    target_aggs = {("A", "B"): 12, ("D",): 3}

    report = ErrorReport(src_aggs, target_aggs)

    assert report.src_aggregates == src_aggs
    assert report.target_aggregates == target_aggs


def test_calc_fabricated():
    src_aggs = {("A", "B"): 10, ("C",): 5}
    target_aggs = {("A", "B"): 12, ("D",): 3, ("E", "F"): 7}

    report = ErrorReport(src_aggs, target_aggs)
    report.calc_fabricated()

    # D and E,F are fabricated (not in source)
    assert report.fabricated_count == 10  # 3 + 7
    assert report.fabricated_count_by_len[1] == 3  # D
    assert report.fabricated_count_by_len[2] == 7  # E,F


def test_calc_suppressed():
    src_aggs = {("A", "B"): 10, ("C",): 5, ("X", "Y", "Z"): 15}
    target_aggs = {("A", "B"): 12}

    report = ErrorReport(src_aggs, target_aggs)
    report.calc_suppressed()

    # C and X,Y,Z are suppressed (not in target)
    assert report.suppressed_count == 20  # 5 + 15
    assert report.suppressed_count_by_len[1] == 5  # C
    assert report.suppressed_count_by_len[3] == 15  # X,Y,Z


def test_calc_mean():
    src_aggs = {("A", "B"): 10, ("C",): 5, ("D", "E"): 20, ("F",): 15}
    target_aggs = {}

    report = ErrorReport(src_aggs, target_aggs)
    report.calc_mean()

    assert report.mean_count == 12.5  # (10 + 5 + 20 + 15) / 4
    assert report.mean_count_by_len[1] == 10.0  # (5 + 15) / 2
    assert report.mean_count_by_len[2] == 15.0  # (10 + 20) / 2


def test_calc_errors():
    src_aggs = {("A", "B"): 10, ("C",): 5, ("D", "E"): 20}
    target_aggs = {("A", "B"): 12, ("C",): 3, ("D", "E"): 25}

    report = ErrorReport(src_aggs, target_aggs)
    report.calc_errors()

    # Errors: |12-10| + |3-5| + |25-20| = 2 + 2 + 5 = 9 total, mean = 3.0
    assert report.mean_error == 3.0
    assert report.mean_error_by_len[1] == 2.0  # C: |3-5|
    assert report.mean_error_by_len[2] == 3.5  # A,B and D,E: (2 + 5) / 2


def test_calc_total():
    aggregates = {("A", "B"): 10, ("C",): 5, ("D", "E", "F"): 15, ("G",): 20}

    total, total_by_len = ErrorReport.calc_total(aggregates)

    assert total == 50  # 10 + 5 + 15 + 20
    assert total_by_len[1] == 25  # C + G
    assert total_by_len[2] == 10  # A,B
    assert total_by_len[3] == 15  # D,E,F


def test_gen_creates_dataframe():
    src_aggs = {("A", "B"): 10, ("C",): 5}
    target_aggs = {("A", "B"): 12, ("D",): 3}

    report = ErrorReport(src_aggs, target_aggs)
    result_df = report.gen()

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == [
        "Length",
        "Count +/- Error",
        "Suppressed %",
        "Fabricated %",
    ]
    # Should have rows for each length + overall row
    assert len(result_df) >= 2


def test_gen_with_multiple_lengths():
    src_aggs = {
        ("A",): 100,
        ("B",): 50,
        ("C", "D"): 200,
        ("E", "F"): 150,
        ("G", "H", "I"): 300,
    }
    target_aggs = {
        ("A",): 95,
        ("B",): 55,
        ("C", "D"): 190,
        ("E", "F"): 160,
        ("G", "H", "I"): 310,
        ("J",): 10,  # Fabricated
    }

    report = ErrorReport(src_aggs, target_aggs)
    result_df = report.gen()

    # Should have rows for lengths 1, 2, 3, and overall
    assert len(result_df) == 4
    assert "1" in result_df["Length"].values
    assert "2" in result_df["Length"].values
    assert "3" in result_df["Length"].values
    assert "Overall" in result_df["Length"].values


def test_gen_empty_aggregates():
    src_aggs = {}
    target_aggs = {}

    report = ErrorReport(src_aggs, target_aggs)
    report.calc_fabricated()
    report.calc_suppressed()

    assert report.fabricated_count == 0
    assert report.suppressed_count == 0


def test_error_report_with_matching_aggregates():
    # Test when source and target are identical
    aggs = {("A", "B"): 10, ("C",): 5}

    report = ErrorReport(aggs, aggs)
    result_df = report.gen()

    # Should have zero fabrication and suppression
    assert "0.00 %" in result_df[result_df["Length"] == "Overall"]["Suppressed %"].values[0]
    assert "0.00 %" in result_df[result_df["Length"] == "Overall"]["Fabricated %"].values[0]


def test_calc_errors_with_no_overlap():
    # Test when there's no overlap between source and target
    src_aggs = {("A",): 10, ("B",): 5}
    target_aggs = {("C",): 7, ("D",): 3}

    report = ErrorReport(src_aggs, target_aggs)
    report.calc_errors()

    # No common keys, so errors list is empty - mean of empty is nan
    assert np.isnan(report.mean_error)
    assert len(report.mean_error_by_len) == 0
