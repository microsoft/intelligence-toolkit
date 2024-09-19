# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import numpy as np
import pandas as pd
import pytest

from toolkit.anonymize_code_data.classes import ErrorReport, calc_total


class TestErrorReport:
    @pytest.fixture()
    def src_aggregates(self) -> dict[tuple[str, str], int]:
        return {
            ("name:Aries", "last_name:Gemini"): 5,
            ("name:Michael", "last_name:Scor"): 7,
            ("name:Lex", "last_name:Luthol"): 8,
            ("name:Peter", "last_name:Pater"): 5,
            ("name:Morgan", "last_name:Blockman"): 1,
            ("name:Robert", "last_name:DeMuro"): 2,
        }

    @pytest.fixture()
    def target_aggregates(self) -> dict[tuple[str, str, str], int]:
        return {
            ("name:Aries", "last_name:Gemini", "age:32"): 1,
            ("name:Michael", "last_name:Scor", "age:37"): 2,
        }

    def test_calc_fabricated(self, src_aggregates, target_aggregates):
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_fabricated()
        assert report.fabricated_count == 3
        assert report.fabricated_count_by_len == {3: 3}

    def test_calc_fabricated_with_mixed_keys(self):
        mixed_src_aggregates = {
            ("name:Aries", "last_name:Gemini", "age:32"): 1,
            ("name:Lex", "age:30"): 2,
            ("name:Joss", "last_name:Water", "age:40"): 3,
        }

        mixed_target_aggregates = {
            ("name:Aries", "last_name:Gemini"): 5,
            ("name:Michael", "last_name:Scor"): 7,
            ("name:Lex"): 8,
        }
        report = ErrorReport(mixed_src_aggregates, mixed_target_aggregates)
        report.calc_fabricated()
        assert report.fabricated_count == 20
        assert report.fabricated_count_by_len == {2: 12, 8: 8}

    def test_calc_fabricated_exists(self, src_aggregates, target_aggregates) -> None:
        src_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        target_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_fabricated()
        assert report.fabricated_count == 3
        assert report.fabricated_count_by_len == {3: 3}

    def test_calc_supressed(self, src_aggregates, target_aggregates) -> None:
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_suppressed()
        assert report.suppressed_count == 28
        assert report.suppressed_count_by_len == {2: 28}

    def test_calc_supressed_exists(self, src_aggregates, target_aggregates) -> None:
        src_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        target_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1

        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_suppressed()
        assert report.suppressed_count == 28
        assert report.suppressed_count_by_len == {2: 28}

    def test_calc_mean(self, src_aggregates, target_aggregates) -> None:
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_mean()
        assert round(report.mean_count, 2) == 4.67
        assert report.mean_count_by_len == {2: 4.666666666666667}

    def test_calc_errors_empty(self, src_aggregates, target_aggregates) -> None:
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_errors()
        assert np.isnan(report.mean_error)
        assert report.mean_error_by_len == {}

    def test_calc_errors_zero(self, src_aggregates, target_aggregates) -> None:
        src_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        target_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_errors()
        assert report.mean_error == 0.0
        assert report.mean_error_by_len == {3: 0.0}

    def test_calc_errors(self, src_aggregates, target_aggregates) -> None:
        src_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        target_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 3
        report = ErrorReport(src_aggregates, target_aggregates)
        report.calc_errors()
        assert report.mean_error == 2.0
        assert report.mean_error_by_len == {3: 2.0}

    def test_gen(self, src_aggregates, target_aggregates) -> None:
        report = ErrorReport(src_aggregates, target_aggregates)
        result_df = report.gen()

        expected_df = pd.DataFrame(
            {
                "Length": ["Overall"],
                "Count +/- Error": ["4.67 +/- nan"],
                "Suppressed %": ["100.00 %"],
                "Fabricated %": ["100.00 %"],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_gen_changed(self, src_aggregates, target_aggregates) -> None:
        src_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 1
        target_aggregates[("name:Alvin", "last_name:Shipminck", "age:42")] = 3
        report = ErrorReport(src_aggregates, target_aggregates)
        result_df = report.gen()

        expected_df = pd.DataFrame(
            {
                "Length": ["3", "Overall"],
                "Count +/- Error": ["1.00 +/- 2.00", "4.14 +/- 2.00"],
                "Suppressed %": ["0.00 %", "96.55 %"],
                "Fabricated %": ["50.00 %", "50.00 %"],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)


class TestCalcTotal:
    def test_calc_total(self):
        src_aggregates = {("a:b"): 1, ("b:c"): 2, ("e:f"): 3}
        total, total_by_len = calc_total(src_aggregates)
        assert total == 6
        assert total_by_len == {3: 6}

    def test_empty_dictionary(self):
        src_aggregates = {}
        total, total_by_len = calc_total(src_aggregates)
        assert total == 0
        assert total_by_len == {}

    def test_multiple_keys_same_length(self) -> None:
        src_aggregates = {("a:b"): 1, ("b:c"): 2, ("e:f"): 3, ("g:h"): 4}
        total, total_by_len = calc_total(src_aggregates)
        assert total == 10
        assert total_by_len == {3: 10}

    def test_negative_zero_values(self) -> None:
        src_aggregates = {("a:b"): 0, ("b:c"): -2, ("e:f", "o:p"): 3}

        total, total_by_len = calc_total(src_aggregates)
        assert total == 1
        assert total_by_len == {2: 3, 3: -2}

    def test_mixed_length_zero(self) -> None:
        src_aggregates = {("a:b"): 0, ("b:c"): 0, ("e:f", "o:p"): 0}
        total, total_by_len = calc_total(src_aggregates)
        assert total == 0
        assert total_by_len == {3: 0, 2: 0}
