# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np

import sys

import pandas as pd

def fix_null_ints(in_df):
    df = in_df.copy()
    for col, dt in zip(df.columns, df.dtypes):
        if dt == 'float64':
            idf = df[[col]].copy()
            idf['float'] = [x if not np.isnan(x) else 0 for x in idf[col]]
            idf['int'] = [int(x) if not np.isnan(x) else 0 for x in idf[col]]
            idf['float_s'] = [x if not np.isnan(x) else -sys.maxsize for x in idf[col]]
            idf['int_s'] = [int(x) if not np.isnan(x) else -sys.maxsize for x in idf[col]]
            fsum = idf['float'].sum()
            isum = idf['int'].sum()
            if int(fsum) == int(isum):
                df[col] = idf['int_s']
                df[col] = df[col].astype('Int64')
                df[col] = df[col].replace(-sys.maxsize, np.nan)
    return df

def get_current_time():
    return pd.Timestamp.now().strftime('%Y%m%d%H%M%S')