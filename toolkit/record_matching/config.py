# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from typing import TypedDict

max_rows_to_show = 1000
entity_label = "Entity"
cache_name = "record_matching"

DEFAULT_COLUMNS_DONT_CONVERT = ["Entity ID", "Entity name", "Dataset"]
DEFAULT_SENTENCE_PAIR_JACCARD_THRESHOLD = 0.75
DEFAULT_MAX_RECORD_DISTANCE = 0.05


class AttributeToMatch(TypedDict):
    label: str
    columns: list[str]