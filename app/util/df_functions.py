# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import math
import re

import pandas as pd
from dateutil import parser as dateparser


def fix_null_ints(in_df: pd.DataFrame) -> pd.DataFrame:
    df = in_df.copy()
    for col, dt in zip(df.columns, df.dtypes, strict=False):
        if dt == "float64":
            try:
                df[col] = df[col].astype("Int64")
                df[col] = df[col].where(pd.notna(df[col]), pd.NA)
                df[col] = df[col].astype(str).replace("<NA>", "")
            except Exception as e:
                print(f"Error converting column {col} to Int64: {e}")

    return df


def get_current_time():
    return pd.Timestamp.now().strftime("%Y%m%d%H%M%S")


def quantize_datetime(input_df, col, bin_size):
    func = None
    def extract_elements(x):
        parts = str(x).split("-")
        if len(parts) == 1:
            parts = str(x).split("/")
        if len(parts) == 1:
            parts = str(x).split(" ")

        elements = {}
        for part in parts:
            if re.match(r"\d{4}", part): # year
                elements['year'] = part
            elif re.match(r"\d{2}", part): # month
                month_cand = int(part)
                if month_cand < 13:
                    elements['month'] = month_cand
                    elements['half'] = "H1" if month_cand < 7 else "H2"
                    elements['quarter'] = (
                        "Q1"
                        if month_cand < 4
                        else "Q2"
                        if month_cand < 7
                        else "Q3"
                        if month_cand < 10
                        else "Q4"
                    )
            elif re.match(r"H1|H2", part):
                elements['half'] = part
                elements['quarter'] = "Q1" if part == "H1" else "Q3" # default to first quarter of half
                elements['month'] = 1 if part == "H1" else 7
            elif re.match(r"Q1|Q2|Q3|Q4", part):
                elements['quarter'] = part
                elements['half'] = "H1" if part == "Q1" or part == "Q2" else "H2"
                elements['month'] = 1 if part == "Q1" else 4 if part == "Q2" else 7 if part == "Q3" else 10
        return elements
    
    if bin_size == "Year":

        def convert(x):
            # parse as datetime using dateutil
            try:
                dt = dateparser.parse(str(x))
                return str(dt.year)
            except:
                elements = extract_elements(x)
                if 'year' in elements:
                    return str(elements['year'])
                else:
                    return ""
        func = convert
    elif bin_size == "Half":
        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                half = "H1" if dt.month < 7 else "H2"
                return str(dt.year) + "-" + half
            except:
                elements = extract_elements(x)
                if 'year' in elements and 'half' in elements:
                    return str(elements['year']) + '-' + str(elements['half'])
                else:
                    return ""
        func = convert
    elif bin_size == "Quarter":
        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                quarter = (
                    "Q1"
                    if dt.month < 4
                    else "Q2"
                    if dt.month < 7
                    else "Q3"
                    if dt.month < 10
                    else "Q4"
                )
                return str(dt.year) + "-" + quarter
            except:
                elements = extract_elements(x)
                if 'year' in elements and 'quarter' in elements:
                    return str(elements['year']) + '-' + str(elements['quarter'])
                else:
                    return ""
        func = convert
    elif bin_size == "Month":

        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                return str(dt.year) + "-" + str(dt.month).zfill(2)
            except:
                elements = extract_elements(x)
                if 'year' in elements and 'month' in elements:
                    # pad with 0s if needed
                    return str(elements['year']) + '-' + str(elements['month']).zfill(2)
                else:
                    return ""
        func = convert
    elif bin_size == "Day":

        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                return (
                    str(dt.year)
                    + "-"
                    + str(dt.month).zfill(2)
                    + "-"
                    + str(dt.day).zfill(2)
                )
            except:
                return ""

        func = convert
    result = input_df[col].apply(func)
    return result


def quantize_numeric(input_df, col, num_bins, trim_percent):
    distinct_values = tuple(
        sorted([x for x in input_df[col].unique() if str(x) != "nan"])
    )

    # print(
    #     f"Quantizing {col} with distinct values: {distinct_values} into {num_bins} bins with {trim_percent} trim percent"
    # )
    if len(distinct_values) < 2:
        return
    if distinct_values == (0, 1):
        return
    sorted_values = sorted([x for x in input_df[col].values if str(x) != "nan"])
    processed_df = input_df.copy(deep=True)
    # first, calculate the top and bottom trim_percent values and apply top and bottom coding
    top = min(
        len(sorted_values) - 1,
        math.floor(len(sorted_values) * (1 - trim_percent)),
    )
    top_trim = sorted_values[top]
    bx = math.floor(len(sorted_values) * trim_percent)
    bottom_trim = sorted_values[bx]
    processed_df.loc[input_df[col] > top_trim, col] = top_trim
    processed_df.loc[input_df[col] < bottom_trim, col] = bottom_trim
    target_bin_width = (top_trim - bottom_trim) / num_bins
    # round target bin width to a multiple of N x 10^k for N = [1, 2, 2.5, 5]. N and k should be chosen to exceed the target bin width by the least positive amount
    k = math.floor(math.log10(target_bin_width))
    n_bin_sizes = {}
    n_excess = {}
    for N in [1, 2, 2.5, 5]:
        n_bin_sizes[N] = N * pow(10, k)
        if n_bin_sizes[N] < target_bin_width:
            n_bin_sizes[N] = N * pow(10, k + 1)
        n_excess[N] = n_bin_sizes[N] - target_bin_width
    # find the N that minimizes the excess
    min_excess_n = sorted(n_excess.items(), key=lambda x: x[1])[0][0]
    selected_bin_size = n_bin_sizes[min_excess_n]
    # next, calculate the bin edges

    lower_bin_edge = (bottom_trim // selected_bin_size) * selected_bin_size
    bin_edges = [lower_bin_edge * 0.999999999999]
    last_bin = lower_bin_edge
    while last_bin < top_trim:
        last_bin += selected_bin_size
        bin_edges.append(last_bin)
    # finally, bin the values
    values, _bins = pd.cut(
        processed_df[col],
        bins=bin_edges,
        retbins=True,
        include_lowest=False,
    )

    results = []
    for v in values:
        if str(v) == "nan":
            results.append("")
        else:
            left = str(v.left)
            right = str(v.right)
            if left.endswith(".0"):
                left = left[:-2]
            if right.endswith(".0"):
                right = right[:-2]
            results.append("(" + left + "-" + right + "]")
    return results
