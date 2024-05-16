# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict

import workflows.attribute_patterns.config as config

class RecordCounter:
    def __init__(self, df):
        self.counter = 0
        self.df = df
        self.periods = sorted(df['Period'].unique())
        self.atts = sorted(df['Full Attribute'].unique())
        att_to_ids_df = df[['Subject ID', 'Full Attribute']].groupby('Full Attribute').agg(list).reset_index()
        self.att_to_ids = dict(zip(att_to_ids_df['Full Attribute'], [set(x) for x in att_to_ids_df['Subject ID']]))
        # do same for Period
        time_to_ids_df = df[['Subject ID', 'Period']].groupby('Period').agg(list).reset_index()
        self.att_to_ids.update(dict(zip(time_to_ids_df['Period'], [set(x) for x in time_to_ids_df['Subject ID']])))
        self.cache = {}

    def count_records(self, atts):
        key = ';'.join(sorted(atts))
        # if key in self.cache.keys():
        #     return self.cache[key]
        # else:
        type_to_vals = defaultdict(list)
        for att in atts:
            type_to_vals[att.split(config.att_val_sep)[0]].append(att)
        ids = set()
        for ix, (typ, vals) in enumerate(type_to_vals.items()):
            combined_atts = set()
            for val in vals:
                combined_atts.update(self.att_to_ids[val])
            if ix == 0:
                ids.update(combined_atts)
            else:
                ids.intersection_update(combined_atts)
        count = len(ids)
        # self.cache[key] = count
        return count

    def compute_period_mean_sd_max(self, atts):
        counts = []
        for p in self.periods:
            counts.append(self.count_records([p] + atts))
        mean = np.mean(counts) if len(counts) > 0 else 0
        sd = np.std(counts) if len(counts) > 0 else 0
        max = np.max(counts) if len(counts) > 0 else 0
        return mean, sd, max
    
    def create_time_series_rows(self, atts):
        rows = []
        for p in self.periods:
            count = self.count_records([p] + atts)
            rows.append([p, ' & '.join(atts), count])
        return rows