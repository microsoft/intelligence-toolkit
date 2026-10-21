# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import math
import re
import sys

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

# Errors dateutil can raise for unparseable or out-of-range date strings
_DATE_PARSE_ERRORS = (ValueError, OverflowError, TypeError)


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


def fix_null_ints_str(in_df: pd.DataFrame) -> pd.DataFrame:
    """Convert float columns to nullable ints rendered as strings, leaving other columns untouched."""
    df = in_df.copy()
    for col, dt in zip(df.columns, df.dtypes, strict=False):
        if dt == "float64":
            try:
                df[col] = df[col].astype("Int64")
                df[col] = df[col].where(pd.notna(df[col]), pd.NA)
                df[col] = df[col].astype(str).replace("<NA>", "")
            except (TypeError, ValueError) as e:
                print(f"Error converting column {col} to Int64: {e}")

    return df


def get_current_time() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d%H%M%S")


def suppress_boolean_binary(
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


def _quarter_of_month(month: int) -> str:
    return "Q1" if month < 4 else "Q2" if month < 7 else "Q3" if month < 10 else "Q4"


def _extract_date_elements(x) -> dict:
    """Best-effort extraction of year/half/quarter/month from a non-standard date string."""
    parts = str(x).split("-")
    if len(parts) == 1:
        parts = str(x).split("/")
    if len(parts) == 1:
        parts = str(x).split(" ")

    elements = {}
    for part in parts:
        if re.match(r"\d{4}", part):  # year
            elements["year"] = part
        elif re.match(r"\d{2}", part):  # month
            month_cand = int(part)
            if month_cand < 13:
                elements["month"] = month_cand
                elements["half"] = "H1" if month_cand < 7 else "H2"
                elements["quarter"] = _quarter_of_month(month_cand)
        elif re.match(r"H1|H2", part):
            elements["half"] = part
            elements["quarter"] = "Q1" if part == "H1" else "Q3"  # default to first quarter of half
            elements["month"] = 1 if part == "H1" else 7
        elif re.match(r"Q1|Q2|Q3|Q4", part):
            elements["quarter"] = part
            elements["half"] = "H1" if part in ("Q1", "Q2") else "H2"
            elements["month"] = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[part]
    return elements


def quantize_datetime(input_df, col, bin_size):
    func = None
    if bin_size == "Year":

        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                return str(dt.year)
            except _DATE_PARSE_ERRORS:
                elements = _extract_date_elements(x)
                return str(elements["year"]) if "year" in elements else ""

        func = convert
    elif bin_size == "Half":

        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                half = "H1" if dt.month < 7 else "H2"
                return str(dt.year) + "-" + half
            except _DATE_PARSE_ERRORS:
                elements = _extract_date_elements(x)
                if "year" in elements and "half" in elements:
                    return str(elements["year"]) + "-" + str(elements["half"])
                return ""

        func = convert
    elif bin_size == "Quarter":

        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                return str(dt.year) + "-" + _quarter_of_month(dt.month)
            except _DATE_PARSE_ERRORS:
                elements = _extract_date_elements(x)
                if "year" in elements and "quarter" in elements:
                    return str(elements["year"]) + "-" + str(elements["quarter"])
                return ""

        func = convert
    elif bin_size == "Month":

        def convert(x):
            try:
                dt = dateparser.parse(str(x))
                return str(dt.year) + "-" + str(dt.month).zfill(2)
            except _DATE_PARSE_ERRORS:
                elements = _extract_date_elements(x)
                if "year" in elements and "month" in elements:
                    return str(elements["year"]) + "-" + str(elements["month"]).zfill(2)
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
            except _DATE_PARSE_ERRORS:
                return ""

        func = convert
    return input_df[col].apply(func)


def quantize_numeric(input_df, col, num_bins, trim_percent):
    distinct_values = tuple(
        sorted([x for x in input_df[col].unique() if str(x) != "nan"])
    )

    if len(distinct_values) < 2:
        return None
    if distinct_values == (0, 1):
        return None
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
