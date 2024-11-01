# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import sys

import numpy as np
import pandas as pd


def fix_null_ints(in_df: pd.DataFrame) -> pd.DataFrame:
    df = in_df.copy()
    for col, dt in zip(df.columns, df.dtypes, strict=False):
        if dt == "float64":
            idf = df[[col]].copy()
            idf["float"] = [x if not np.isnan(x) else 0 for x in idf[col]]
            idf["int"] = [int(x) if not np.isnan(x) else 0 for x in idf[col]]
            idf["float_s"] = [x if not np.isnan(x) else -sys.maxsize for x in idf[col]]
            idf["int_s"] = [
                int(x) if not np.isnan(x) else -sys.maxsize for x in idf[col]
            ]
            fsum = idf["float"].sum()
            isum = idf["int"].sum()
            if int(fsum) == int(isum):
                df[col] = idf["int_s"]
                df[col] = df[col].astype("Int64")
                df[col] = df[col].replace(-sys.maxsize, np.nan)

    return df.astype(str).replace("nan", "").replace("<NA>", "")


def get_current_time() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

def supress_boolean_binary(
    input_df: pd.DataFrame, output_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    if output_df is None:
        output_df = input_df.copy()
    for col in input_df.columns:
        unique_values = [str(x) for x in input_df[col].unique()]
        is_three_with_none = len(unique_values) == 3 and input_df[col].isna().any()
        if len(unique_values) <= 2 or is_three_with_none:
            if "0" in unique_values or "0.0" in unique_values:
                output_df[col] = (
                    input_df[col]
                    .astype(str)
                    .replace("0", np.nan)
                    .replace("0.0", np.nan)
                )
            elif "False" in unique_values:
                output_df[col] = input_df[col].astype(str).replace("False", np.nan)
    return output_df