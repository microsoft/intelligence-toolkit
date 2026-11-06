# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Entity record matching: unify, embed, block, score, cluster, evaluate."""

from itk_match.matcher import Matcher
from itk_match.models import (
    Dataset,
    Group,
    GroupEvaluation,
    MatchConfig,
    MatchResult,
    Pair,
)

__all__ = [
    "Dataset",
    "Group",
    "GroupEvaluation",
    "MatchConfig",
    "MatchResult",
    "Matcher",
    "Pair",
]
