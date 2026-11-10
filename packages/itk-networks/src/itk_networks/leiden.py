# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Hierarchical Leiden community detection through graspologic.

Importing this module requires the ``leiden`` extra (``itk-networks[leiden]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graspologic.partition import hierarchical_leiden

if TYPE_CHECKING:
    import networkx as nx


class GraspologicLeiden:
    """:class:`CommunityDetector` matching the legacy ``get_subgraph`` call."""

    def __init__(self, resolution: float = 1.0) -> None:
        self.resolution = resolution

    def detect(self, graph: nx.Graph, max_size: int, seed: int) -> dict[str, int]:
        clustering = hierarchical_leiden(
            graph,
            resolution=self.resolution,
            random_seed=seed,
            max_cluster_size=max_size,
        ).final_level_hierarchical_clustering()
        return {str(node): int(cluster) for node, cluster in clustering.items()}
