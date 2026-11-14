# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Estimate how well a categorical table can be synthesized."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from itk_anonymize.models import SynthesizabilityStatistics
from itk_anonymize.prepare import EMPTY, prepare_frame

if TYPE_CHECKING:
    import polars as pl

_MISSING = {EMPTY, "nan", "NaN"}


def analyze_synthesizability(frame: pl.DataFrame) -> SynthesizabilityStatistics:
    """Compare the attribute combination space with the number of records.

    Reproduces the legacy ``analyze_synthesizability`` arithmetic, including
    rounding combinations per row to one decimal.
    """
    if frame.height == 0:
        msg = "Cannot analyze an empty frame"
        raise ValueError(msg)
    text = prepare_frame(frame)
    distinct_counts: list[int] = []
    non_empty_cells = 0
    for name in text.columns:
        column = text.get_column(name)
        values = [v for v in column.unique().to_list() if v not in _MISSING]
        if values:
            distinct_counts.append(len(values))
        non_empty_cells += int((~column.is_in(list(_MISSING))).sum())
    distinct_counts.sort()
    overall_att_count = sum(distinct_counts)
    possible_combinations = math.prod(distinct_counts)
    possible_combinations_per_row = round(possible_combinations / frame.height, 1)
    mean_vals_per_record = non_empty_cells / frame.height
    max_combinations_per_record = 2**mean_vals_per_record
    return SynthesizabilityStatistics(
        num_cols=len(text.columns),
        overall_att_count=overall_att_count,
        possible_combinations=possible_combinations,
        possible_combinations_per_row=possible_combinations_per_row,
        mean_vals_per_record=mean_vals_per_record,
        max_combinations_per_record=max_combinations_per_record,
        excess_combinations_ratio=possible_combinations_per_row
        / max_combinations_per_record,
    )
