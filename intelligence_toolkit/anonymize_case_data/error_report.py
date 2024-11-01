# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

from collections import defaultdict
import numpy as np
import pandas as pd


class ErrorReport:
    def __init__(self, src_aggregates, target_aggregates):
        self.src_aggregates = src_aggregates
        self.target_aggregates = target_aggregates

    def calc_fabricated(self):
        self.fabricated_count = 0
        self.fabricated_count_by_len = defaultdict(int)

        for u, v in self.target_aggregates.items():
            if u not in self.src_aggregates:
                self.fabricated_count += v
                self.fabricated_count_by_len[len(u)] += v

    def calc_suppressed(self):
        self.suppressed_count = 0
        self.suppressed_count_by_len = defaultdict(int)

        for o, v in self.src_aggregates.items():
            if o not in self.target_aggregates:
                self.suppressed_count += v
                self.suppressed_count_by_len[len(o)] += v

    def calc_mean(self):
        mean = []
        mean_by_len = defaultdict(list)

        for o in self.src_aggregates:
            mean.append(self.src_aggregates[o])
            mean_by_len[len(o)].append(self.src_aggregates[o])

        self.mean_count = np.mean(mean)
        self.mean_count_by_len = {l: np.mean(mean_by_len[l]) for l in mean_by_len}

    def calc_errors(self):
        errors = []
        errors_by_len = defaultdict(list)

        for o in self.src_aggregates:
            if o in self.target_aggregates:
                err = abs(self.target_aggregates[o] - self.src_aggregates[o])
                errors.append(err)
                errors_by_len[len(o)].append(err)

        self.mean_error = np.mean(errors)
        self.mean_error_by_len = {l: np.mean(errors_by_len[l]) for l in errors_by_len}

    def calc_total(aggregates):
        total = 0
        total_by_len = defaultdict(int)

        for k, v in aggregates.items():
            total += v
            total_by_len[len(k)] += v

        return (total, total_by_len)

    def gen(self):
        self.calc_fabricated()
        self.calc_suppressed()
        self.calc_mean()
        self.calc_errors()
        self.src_total, self.src_total_by_len = ErrorReport.calc_total(
            self.src_aggregates
        )
        self.target_total, self.target_total_by_len = ErrorReport.calc_total(
            self.target_aggregates
        )

        rows = [
            [
                str(l),
                f"{self.mean_count_by_len[l]:.2f} +/- {self.mean_error_by_len[l]:.2f}",
                f"{self.suppressed_count_by_len[l] * 100.0 / self.src_total_by_len[l]:.2f} %",
                f"{self.fabricated_count_by_len[l] * 100.0 / self.target_total_by_len[l]:.2f} %",
            ]
            for l in sorted(self.mean_error_by_len.keys())
        ]
        rows.append([
            "Overall",
            f"{self.mean_count:.2f} +/- {self.mean_error:.2f}",
            f"{self.suppressed_count * 100.0 / self.src_total:.2f} %",
            f"{self.fabricated_count * 100.0 / self.target_total:.2f} %",
        ])

        return pd.DataFrame(
            rows,
            columns=[
                "Length",
                "Count +/- Error",
                "Suppressed %",
                "Fabricated %",
            ],
        )
