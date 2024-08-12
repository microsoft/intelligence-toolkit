# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import re

import pandas as pd


def clean_text(text: str | int) -> str:
    # remove punctuation but retain characters and digits in any language
    # compress whitespace to single space
    cleaned_text = re.sub(r"[^\w\s&@\+]", "", str(text)).strip()
    # cleaned_text = re.sub(r"[^\w\s&@+/]", "", str(text)).strip()
    return re.sub(r"\s+", " ", cleaned_text)


def format_data_columns(
    values_df: pd.DataFrame, columns_to_link: list[str], entity_id_column: str | int
) -> pd.DataFrame:
    values_df[entity_id_column] = values_df[entity_id_column].apply(clean_text)
    for value_col in columns_to_link:
        values_df[value_col] = values_df[value_col].apply(clean_text)
    return values_df
