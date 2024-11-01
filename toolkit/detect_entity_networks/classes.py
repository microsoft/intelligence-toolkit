# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from enum import Enum


class FlagAggregatorType(Enum):
    Instance = "Instance"
    Count = "Count"

class SummaryData:
    def __init__(self, entities, attributes, flags, groups, links) -> None:
        self.entities = entities
        self.attributes = attributes
        self.flags = flags
        self.groups = groups
        self.links = links
