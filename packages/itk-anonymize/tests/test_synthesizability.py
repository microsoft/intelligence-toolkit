# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import polars as pl
import pytest
from itk_anonymize.synthesizability import analyze_synthesizability


def test_matches_legacy_arithmetic() -> None:
    frame = pl.DataFrame({"Color": ["Red", "Blue"], "Size": ["Large", "Small"]})
    stats = analyze_synthesizability(frame)
    assert stats.num_cols == 2
    assert stats.overall_att_count == 4
    assert stats.possible_combinations == 4
    assert stats.possible_combinations_per_row == 2.0
    assert stats.mean_vals_per_record == 2.0
    assert stats.max_combinations_per_record == 4.0
    assert stats.excess_combinations_ratio == 0.5


def test_empty_and_null_cells_are_not_values() -> None:
    frame = pl.DataFrame({"A": ["X", "Y", None], "B": ["1", "", ""], "C": ["", "", ""]})
    stats = analyze_synthesizability(frame)
    assert stats.num_cols == 3
    # C has no values, so it contributes nothing to the product
    assert stats.overall_att_count == 3
    assert stats.possible_combinations == 2
    assert stats.mean_vals_per_record == 1.0


def test_single_valued_columns() -> None:
    frame = pl.DataFrame({"Color": ["Red"] * 3, "Size": ["Large"] * 3})
    assert analyze_synthesizability(frame).possible_combinations == 1


def test_per_row_combinations_are_rounded_to_one_decimal() -> None:
    frame = pl.DataFrame({"A": ["1", "2", "3"], "B": ["x", "y", "x"]})
    # 3 * 2 = 6 combinations over 3 rows
    assert analyze_synthesizability(frame).possible_combinations_per_row == 2.0
    frame = pl.DataFrame({"A": ["1", "2", "3", "4", "5", "6", "7"], "B": ["x"] * 7})
    assert analyze_synthesizability(frame).possible_combinations_per_row == 1.0


def test_numeric_columns_are_treated_as_text_categories() -> None:
    frame = pl.DataFrame({"n": [1.0, 2.0, None]})
    stats = analyze_synthesizability(frame)
    assert stats.overall_att_count == 2
    assert stats.mean_vals_per_record == pytest.approx(2 / 3)


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        analyze_synthesizability(pl.DataFrame({"a": []}))
