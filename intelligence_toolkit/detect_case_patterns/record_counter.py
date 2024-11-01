# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from collections import defaultdict

import numpy as np

from intelligence_toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR


class RecordCounter:
    def __init__(self, df):
        self.counter = 0
        self.df = df
        self.periods = sorted(df["Period"].unique())
        self.atts = sorted(df["Full Attribute"].unique())
        att_to_ids_df = (
            df[["Subject ID", "Full Attribute"]]
            .groupby("Full Attribute")
            .agg(list)
            .reset_index()
        )
        self.att_to_ids = dict(
            zip(
                att_to_ids_df["Full Attribute"],
                [set(x) for x in att_to_ids_df["Subject ID"]],
                strict=False,
            )
        )
        # do same for Period
        time_to_ids_df = (
            df[["Subject ID", "Period"]].groupby("Period").agg(list).reset_index()
        )
        self.att_to_ids.update(
            dict(
                zip(
                    time_to_ids_df["Period"],
                    [set(x) for x in time_to_ids_df["Subject ID"]],
                    strict=False,
                )
            )
        )
        self.cache = {}

    def count_records(self, atts):
        key = ";".join(sorted(atts))
        if key in self.cache:
            return self.cache[key]

        type_to_vals = defaultdict(list)
        for att in atts:
            type_to_vals[att.split(ATTRIBUTE_VALUE_SEPARATOR)[0]].append(att)
        ids = set()
        for ix, (_typ, vals) in enumerate(type_to_vals.items()):
            combined_atts = set()
            for val in vals:
                combined_atts.update(self.att_to_ids[val])
            if ix == 0:
                ids.update(combined_atts)
            else:
                ids.intersection_update(combined_atts)
        count = len(ids)
        self.cache[key] = count
        return count

    def compute_period_mean_sd_max(self, atts):
        counts = []
        for p in self.periods:
            counts.append(self.count_records([p, *atts]))
        np_mean = np.mean(counts) if len(counts) > 0 else 0
        np_sd = np.std(counts) if len(counts) > 0 else 0
        np_max = np.max(counts) if len(counts) > 0 else 0
        return np_mean, np_sd, np_max

    def create_time_series_rows(self, atts):
        rows = []
        for p in self.periods:
            count = self.count_records([p, *atts])
            rows.append([p, " & ".join(atts), count])
        return rows
