# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
from enum import Enum

from python.helpers.constants import CACHE_PATH

list_sep = ";"
max_rows_to_show = 1000
entity_label = "ENTITY"
cache_dir = os.path.join(CACHE_PATH, "risk_networks")
outputs_dir = os.path.join(cache_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)


class LinkType(Enum):
    EntityAttribute = "Entity-Attribute"
    EntityFlag = "Entity-Flag"
    EntityGroup = "Entity-Group"


class AttributeColumnType(Enum):
    ColumnName = "Use column name"
    CustomName = "Use custom name"
