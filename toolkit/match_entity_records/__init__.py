# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import os

from .api import MatchEntityRecords
from .classes import AttributeToMatch, RecordsModel
from .prepare_model import (
    build_attribute_options,
    build_attributes_list,
    format_model_df,
)


def get_readme() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


__all__ = [
    "AttributeToMatch",
    "MatchEntityRecords",
    "RecordsModel",
    "build_attribute_options",
    "build_attributes_list",
    "format_model_df",
    "get_readme",
]
