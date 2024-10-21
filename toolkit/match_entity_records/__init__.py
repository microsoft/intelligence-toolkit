# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import os

from .api import MatchEntityRecords
from .prepare_model import build_attribute_options, format_dataset


def get_readme() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


__all__ = [
    "MatchEntityRecords",
    "get_readme",
    "format_dataset",
    "build_attribute_options",
]
