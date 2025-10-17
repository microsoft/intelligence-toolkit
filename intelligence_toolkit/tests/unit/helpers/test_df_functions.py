# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import sys

import numpy as np
import pandas as pd
import pytest

from intelligence_toolkit.helpers.df_functions import (
    fix_null_ints,
    get_current_time,
    suppress_boolean_binary,
)


def test_fix_null_ints_with_float_integers():
    df = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0],
        "col2": ["a", "b", "c"]
    })
    
    result = fix_null_ints(df)
    
    assert result["col1"].dtype == "object"  # Converted to string
    assert result["col1"].tolist() == ["1", "2", "3"]


def test_fix_null_ints_with_nan_values():
    df = pd.DataFrame({
        "col1": [1.0, np.nan, 3.0],
        "col2": ["a", "b", "c"]
    })
    
    result = fix_null_ints(df)
    
    assert result["col1"].tolist() == ["1", "", "3"]


def test_fix_null_ints_with_mixed_floats():
    df = pd.DataFrame({
        "col1": [1.5, 2.5, 3.5],
        "col2": ["a", "b", "c"]
    })
    
    result = fix_null_ints(df)
    
    # Should not convert non-integer floats
    assert "1.5" in result["col1"].tolist()


def test_fix_null_ints_does_not_modify_original():
    df = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0],
        "col2": ["a", "b", "c"]
    })
    original_df = df.copy()
    
    fix_null_ints(df)
    
    pd.testing.assert_frame_equal(df, original_df)


def test_fix_null_ints_empty_dataframe():
    df = pd.DataFrame()
    result = fix_null_ints(df)
    
    assert len(result) == 0


def test_fix_null_ints_no_float_columns():
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    result = fix_null_ints(df)
    
    assert result["col1"].tolist() == ["1", "2", "3"]
    assert result["col2"].tolist() == ["a", "b", "c"]


def test_get_current_time():
    result = get_current_time()
    
    assert isinstance(result, str)
    assert len(result) == 14  # YYYYMMDDHHMMSS format
    assert result.isdigit()


def test_get_current_time_format():
    result = get_current_time()
    
    # Should be in YYYYMMDDHHMMSS format
    year = int(result[:4])
    month = int(result[4:6])
    day = int(result[6:8])
    
    assert 2020 <= year <= 2100
    assert 1 <= month <= 12
    assert 1 <= day <= 31


def test_suppress_boolean_binary_with_zeros():
    df = pd.DataFrame({
        "col1": [0, 1, 0, 1],
        "col2": ["a", "b", "c", "d"]
    })
    
    result = suppress_boolean_binary(df)
    
    # Zeros should be converted to NaN
    assert pd.isna(result["col1"].iloc[0])
    assert pd.isna(result["col1"].iloc[2])
    assert result["col1"].iloc[1] == "1"


def test_suppress_boolean_binary_with_floats():
    df = pd.DataFrame({
        "col1": [0.0, 1.0, 0.0, 1.0],
        "col2": ["a", "b", "c", "d"]
    })
    
    result = suppress_boolean_binary(df)
    
    assert pd.isna(result["col1"].iloc[0])
    assert pd.isna(result["col1"].iloc[2])


def test_suppress_boolean_binary_with_false():
    df = pd.DataFrame({
        "col1": [False, True, False, True],
        "col2": ["a", "b", "c", "d"]
    })
    
    result = suppress_boolean_binary(df)
    
    assert pd.isna(result["col1"].iloc[0])
    assert pd.isna(result["col1"].iloc[2])


def test_suppress_boolean_binary_with_three_values_including_nan():
    df = pd.DataFrame({
        "col1": [0, 1, np.nan, 1],
        "col2": ["a", "b", "c", "d"]
    })
    
    result = suppress_boolean_binary(df)
    
    # Should still suppress zeros even with NaN present
    assert pd.isna(result["col1"].iloc[0])


def test_suppress_boolean_binary_preserves_non_binary():
    df = pd.DataFrame({
        "col1": [0, 1, 2, 3],
        "col2": ["a", "b", "c", "d"]
    })
    
    result = suppress_boolean_binary(df)
    
    # Function doesn't convert to string, preserves types when not binary
    assert result["col1"].tolist() == [0, 1, 2, 3]


def test_suppress_boolean_binary_with_output_df():
    input_df = pd.DataFrame({
        "col1": [0, 1, 0, 1],
        "col2": ["a", "b", "c", "d"]
    })
    output_df = pd.DataFrame({
        "col1": ["x", "y", "z", "w"],
        "col2": ["e", "f", "g", "h"]
    })
    
    result = suppress_boolean_binary(input_df, output_df)
    
    # Should modify output_df based on input_df
    assert pd.isna(result["col1"].iloc[0])


def test_suppress_boolean_binary_does_not_modify_original():
    df = pd.DataFrame({
        "col1": [0, 1, 0, 1],
        "col2": ["a", "b", "c", "d"]
    })
    original_df = df.copy()
    
    suppress_boolean_binary(df)
    
    pd.testing.assert_frame_equal(df, original_df)
