# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Node naming and text cleaning shared by every stage.

Graph nodes are strings of the form ``<type>==<value>``; entities use the
reserved type ``ENTITY``. These helpers are the only place that encoding lives.
"""

from __future__ import annotations

import re

ENTITY_LABEL = "ENTITY"
ATTRIBUTE_VALUE_SEPARATOR = "=="
LIST_SEPARATOR = ";"
EQUALITY_EDGE_TYPE = "equality"

_CSV_UNSAFE = re.compile(r"[^\w\s&@\+]")
_WHITESPACE = re.compile(r"\s+")
_ENTITY_PREFIX = f"{ENTITY_LABEL}{ATTRIBUTE_VALUE_SEPARATOR}"


def clean_text(text: str | float) -> str:
    """Drop punctuation (keeping ``& @ +``) and collapse whitespace.

    Identical to the legacy ``clean_text`` so node names, and therefore
    community ids, match the Streamlit workflow.
    """
    cleaned = _CSV_UNSAFE.sub("", str(text)).strip()
    return _WHITESPACE.sub(" ", cleaned)


def qualify(node_type: str, value: str) -> str:
    return f"{node_type}{ATTRIBUTE_VALUE_SEPARATOR}{value}"


def qualify_entity(entity: str) -> str:
    return f"{_ENTITY_PREFIX}{entity}"


def unqualify_entity(node: str) -> str:
    """Strip one leading ``ENTITY==`` prefix, as the legacy frames did."""
    return node.replace(_ENTITY_PREFIX, "", 1)


def is_entity(node: str) -> bool:
    return node.startswith(ENTITY_LABEL)


def node_type(node: str) -> str:
    return node.split(ATTRIBUTE_VALUE_SEPARATOR)[0]


def node_value(node: str) -> str:
    """The value half of a node; the node itself when it has no separator."""
    parts = node.split(ATTRIBUTE_VALUE_SEPARATOR, 1)
    return parts[1] if len(parts) > 1 else node
