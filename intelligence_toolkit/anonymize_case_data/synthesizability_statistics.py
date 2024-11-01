# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

class SynthesizabilityStatistics:
    def __init__(
        self,
        num_cols: int,
        overall_att_count: int,
        possible_combinations: int,
        possible_combinations_per_row: float,
        mean_vals_per_record: float,
        max_combinations_per_record: float,
        excess_combinations_ratio: float
    ) -> None:
        self.num_cols = num_cols
        self.overall_att_count = overall_att_count
        self.possible_combinations = possible_combinations
        self.possible_combinations_per_row = possible_combinations_per_row
        self.mean_vals_per_record = mean_vals_per_record
        self.max_combinations_per_record = max_combinations_per_record
        self.excess_combinations_ratio = excess_combinations_ratio

    def __repr__(self) -> str:
        return f"SynthesizabilityStatistics(num_cols={self.num_cols}, overall_att_count={self.overall_att_count}, possible_combinations={self.possible_combinations}, possible_combinations_per_row={self.possible_combinations_per_row}, mean_vals_per_record={self.mean_vals_per_record}, max_combinations_per_record={self.max_combinations_per_record}, excess_combinations_ratio={self.excess_combinations_ratio})"