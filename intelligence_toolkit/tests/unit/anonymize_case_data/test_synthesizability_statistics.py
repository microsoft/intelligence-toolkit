# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import pytest
from intelligence_toolkit.anonymize_case_data.synthesizability_statistics import (
    SynthesizabilityStatistics,
)


def test_synthesizability_statistics_initialization():
    stats = SynthesizabilityStatistics(
        num_cols=5,
        overall_att_count=100,
        possible_combinations=1000,
        possible_combinations_per_row=10.5,
        mean_vals_per_record=3.2,
        max_combinations_per_record=9.2,
        excess_combinations_ratio=1.14,
    )

    assert stats.num_cols == 5
    assert stats.overall_att_count == 100
    assert stats.possible_combinations == 1000
    assert stats.possible_combinations_per_row == 10.5
    assert stats.mean_vals_per_record == 3.2
    assert stats.max_combinations_per_record == 9.2
    assert stats.excess_combinations_ratio == 1.14


def test_synthesizability_statistics_repr():
    stats = SynthesizabilityStatistics(
        num_cols=3,
        overall_att_count=50,
        possible_combinations=500,
        possible_combinations_per_row=5.0,
        mean_vals_per_record=2.5,
        max_combinations_per_record=5.7,
        excess_combinations_ratio=0.88,
    )

    repr_str = repr(stats)

    assert "SynthesizabilityStatistics" in repr_str
    assert "num_cols=3" in repr_str
    assert "overall_att_count=50" in repr_str
    assert "possible_combinations=500" in repr_str
    assert "possible_combinations_per_row=5.0" in repr_str
    assert "mean_vals_per_record=2.5" in repr_str
    assert "max_combinations_per_record=5.7" in repr_str
    assert "excess_combinations_ratio=0.88" in repr_str


def test_synthesizability_statistics_zero_values():
    stats = SynthesizabilityStatistics(
        num_cols=0,
        overall_att_count=0,
        possible_combinations=0,
        possible_combinations_per_row=0.0,
        mean_vals_per_record=0.0,
        max_combinations_per_record=0.0,
        excess_combinations_ratio=0.0,
    )

    assert stats.num_cols == 0
    assert stats.overall_att_count == 0
    assert stats.possible_combinations == 0
    assert stats.possible_combinations_per_row == 0.0
    assert stats.mean_vals_per_record == 0.0
    assert stats.max_combinations_per_record == 0.0
    assert stats.excess_combinations_ratio == 0.0
