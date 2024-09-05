# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from enum import Enum
from typing import TypedDict

SIMILARITY_THRESHOLD_MIN = 0.001
SIMILARITY_THRESHOLD_MAX = 1.0
DEFAULT_MAX_ATTRIBUTE_DEGREE = 10
ENTITY_LABEL = "ENTITY"
LIST_SEPARATOR = ";"

cache_name = "risk_networks"


class LinkType(Enum):
    EntityAttribute = "Entity-Attribute"
    EntityFlag = "Entity-Flag"
    EntityGroup = "Entity-Group"


class FlagAggregatorType(Enum):
    Instance = "Instance"
    Count = "Count"


class FlagsSummary(TypedDict):
    direct: int
    indirect: int
    paths: int
    entities: int


class NodeFlag(TypedDict):
    node: str
    flags: int | None
    entities: int | None


class Steps(TypedDict):
    source: NodeFlag
    target: NodeFlag
