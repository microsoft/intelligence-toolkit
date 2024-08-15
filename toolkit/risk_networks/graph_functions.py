# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from collections import defaultdict
from typing import Any

import networkx as nx

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback
from toolkit.risk_networks.constants import (
    SIMILARITY_THRESHOLD_MAX,
    SIMILARITY_THRESHOLD_MIN,
)

from . import config


def _merge_condition(x, y) -> bool:
    """
    Merge condition function for merging nodes in the graph.
    """
    x_parts = set(x.split(config.list_sep))
    y_parts = set(y.split(config.list_sep))
    return any(
        x_part.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
        == y_part.split(ATTRIBUTE_VALUE_SEPARATOR)[i]
        for i in range(2)
        for x_part in x_parts
        for y_part in y_parts
    )


def _merge_node_list(G, merge_list) -> nx.Graph:  # noqa: N803
    G1 = G.copy()
    merged_node = config.list_sep.join(sorted(merge_list))
    merged_type = config.list_sep.join(sorted([G.nodes[n]["type"] for n in merge_list]))
    merged_risk = max(G.nodes[n]["flags"] for n in merge_list)
    G1.add_node(merged_node, type=merged_type, flags=merged_risk)
    for n in merge_list:
        for nn in G.neighbors(n):
            if nn not in merge_list:
                G1.add_edge(merged_node, nn)
        G1.remove_node(n)
    return G1


def _merge_nodes(G, merge_condition=_merge_condition) -> nx.Graph:  # noqa: N803
    nodes = list(G.nodes())  # may change during iteration
    for node in nodes:
        if node not in G.nodes():
            continue
        neighbours = list(G.neighbors(node))
        merge_list = [node]
        for n in neighbours:
            if n not in G.nodes():
                continue
            if merge_condition(node, n):
                merge_list.append(n)
        if len(merge_list) > 1:
            G = _merge_node_list(G, merge_list)

    return G


def simplify_graph(C) -> nx.Graph:  # noqa: N803
    S = C.copy()
    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    S = _merge_nodes(S)

    # remove single degree attributes
    for node in list(S.nodes()):
        if S.degree(node) < 2 and not node.startswith(config.entity_label):
            S.remove_node(node)

    return S
