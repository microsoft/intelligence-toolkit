import numpy as np

import sys

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
                print(f'Column {col} is already int-compatible')
                df[col] = idf['int_s']
                df[col] = df[col].astype('Int64')
                df[col] = df[col].replace(-sys.maxsize, np.nan)
    return df