# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import re


def clean_text_for_csv(text: str | int) -> str:
    # Replace non-alphanumeric characters
    return re.sub(r"[^\w\s&@\+]", "", str(text))


def clean_for_column_name(text: str | int) -> str:
    # Replace non-alphanumeric characters
    return re.sub(r"[^\w\s&()\-_\+]", "", str(text))