# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs the pipeline end to end, one stage at a time."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl

from itk_networks.communities import find_networks
from itk_networks.embed import embed_nodes
from itk_networks.explore import network_subgraph, simplify, subgraph_frames
from itk_networks.exposure import exposure_data
from itk_networks.models import (
    ATTRIBUTE_LINK_SCHEMA,
    FLAG_LINK_SCHEMA,
    GROUP_LINK_SCHEMA,
    NetworkConfig,
    NetworkResult,
)
from itk_networks.neighbors import infer_links, nearest_neighbors, symmetric
from itk_networks.prepare import (
    attribute_links,
    build_graph,
    flag_links,
    group_links,
    integrate_flags,
)
from itk_networks.project import project_entity_graph, trim_attributes
from itk_networks.records import entity_records
from itk_networks.report import (
    build_report_messages,
    generate_report,
    report_variables,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx
    from itk_core.classes import LLMCallback
    from itk_core.openai_configuration import OpenAIConfiguration
    from itk_core.progress import ProgressBatchCallback

    from itk_networks.communities import CommunityDetector
    from itk_networks.embed import Embedder
    from itk_networks.models import (
        AttributeTable,
        Exposure,
        FlagTable,
        GroupTable,
    )
    from itk_networks.neighbors import NeighborIndex
    from itk_networks.report import ChatFn


def _concat(frames: list[pl.DataFrame], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


class NetworkDetector:
    """Holds the engines; every stage takes a result and returns a new one."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        community_detector: CommunityDetector | None = None,
        chat: ChatFn | None = None,
        neighbor_index: NeighborIndex | None = None,
    ) -> None:
        self.embedder = embedder
        self.community_detector = community_detector
        self.chat = chat
        self.neighbor_index = neighbor_index

    @classmethod
    def from_openai(
        cls,
        configuration: OpenAIConfiguration,
        local_embeddings: bool = False,
        community_detector: CommunityDetector | None = None,
    ) -> NetworkDetector:
        from itk_core.client import OpenAIClient
        from itk_core.openai_embedder import OpenAIEmbedder

        if local_embeddings:
            from itk_core.local_embedder import LocalEmbedder

            embedder: Embedder = LocalEmbedder()
        else:
            embedder = OpenAIEmbedder(configuration)
        return cls(
            embedder,
            community_detector,
            OpenAIClient(configuration).generate_chat_async,
        )

    # -- stage 1 -------------------------------------------------------------

    @staticmethod
    def prepare(
        attributes: Sequence[AttributeTable],
        flags: Sequence[FlagTable] = (),
        groups: Sequence[GroupTable] = (),
    ) -> NetworkResult:
        """Build the attribute graph, the flags frame and the group links."""
        links = _concat([attribute_links(t) for t in attributes], ATTRIBUTE_LINK_SCHEMA)
        flag_frame = _concat([flag_links(t) for t in flags], FLAG_LINK_SCHEMA)
        integrated, max_flags, mean_flags = integrate_flags(flag_frame)
        return NetworkResult(
            graph=build_graph(links),
            attribute_links=links,
            flag_links=flag_frame,
            group_links=_concat([group_links(t) for t in groups], GROUP_LINK_SCHEMA),
            flags=integrated,
            max_entity_flags=max_flags,
            mean_flagged_flags=mean_flags,
        )

    # -- stages 2 and 3 ------------------------------------------------------

    async def index(
        self,
        result: NetworkResult,
        node_types: list[str],
        neighbors: int = 20,
        callbacks: list[ProgressBatchCallback] | None = None,
        cache: bool = True,
    ) -> NetworkResult:
        """Embed nodes of the given types and store their nearest neighbours."""
        if self.embedder is None:
            msg = "NetworkDetector has no embedder; pass one or use from_openai"
            raise ValueError(msg)
        texts, embeddings = await embed_nodes(
            result.graph, node_types, self.embedder, callbacks, cache
        )
        distances, indices = nearest_neighbors(
            embeddings, self.neighbor_index, neighbors
        )
        return replace(
            result,
            indexed_node_types=tuple(node_types),
            indexed_texts=tuple(texts),
            embeddings=embeddings,
            neighbor_distances=distances,
            neighbor_indices=indices,
            inferred_links={},
            similarity_threshold=None,
        )

    @staticmethod
    def infer(
        result: NetworkResult,
        similarity_threshold: float,
        callbacks: list[ProgressBatchCallback] | None = None,
    ) -> NetworkResult:
        """Infer links between indexed nodes; no re-embedding."""
        if result.neighbor_distances is None or result.neighbor_indices is None:
            msg = "NetworkResult has not been indexed"
            raise ValueError(msg)
        links = infer_links(
            result.indexed_texts,
            result.neighbor_distances,
            result.neighbor_indices,
            similarity_threshold,
            callbacks,
        )
        return replace(
            result, inferred_links=links, similarity_threshold=similarity_threshold
        )

    @staticmethod
    def clear_inferred(result: NetworkResult) -> NetworkResult:
        return replace(result, inferred_links={}, similarity_threshold=None)

    # -- stage 4 -------------------------------------------------------------

    def identify(
        self, result: NetworkResult, config: NetworkConfig | None = None
    ) -> NetworkResult:
        """Trim, project, find networks and compute entity records."""
        config = config or NetworkConfig()
        links = symmetric(result.inferred_links)
        trimmed_frame, trimmed = trim_attributes(
            result.graph, config.max_attribute_degree, config.removed_attributes
        )
        projected = project_entity_graph(
            result.graph, trimmed, links, config.supporting_attribute_types
        )
        networks, membership = find_networks(projected, config, self.community_detector)
        return replace(
            result,
            config=config,
            inferred_links=links,
            trimmed_attributes=trimmed_frame,
            trimmed_nodes=trimmed,
            networks=networks,
            membership=membership,
            entity_records=entity_records(networks, result.flags, links),
        )

    # -- stages 5 and 6 ------------------------------------------------------

    @staticmethod
    def explore(result: NetworkResult, network_id: int) -> nx.Graph:
        """The attribute subgraph around one identified network."""
        return network_subgraph(
            result.graph,
            result.network(network_id).entities,
            result.membership,
            result.flags,
            result.trimmed_nodes,
            result.inferred_links,
        )

    @staticmethod
    def exposure(result: NetworkResult, network_id: int, entity: str) -> Exposure:
        """Flag exposure of ``entity`` (unqualified id) within one network."""
        subgraph = NetworkDetector.explore(result, network_id)
        return exposure_data(
            result.flags,
            result.network(network_id).entities,
            entity,
            subgraph,
            result.inferred_links,
        )

    @staticmethod
    def report_messages(
        result: NetworkResult,
        network_id: int,
        entity: str = "",
        instructions: str | None = None,
    ) -> list[dict[str, str]]:
        """The prompt the report is generated from, for inspection or a custom LLM."""
        nodes, edges = subgraph_frames(
            simplify(NetworkDetector.explore(result, network_id))
        )
        exposure_markdown = (
            NetworkDetector.exposure(result, network_id, entity).markdown
            if entity
            else ""
        )
        variables = report_variables(
            nodes,
            edges,
            network_id,
            entity,
            exposure_markdown,
            result.max_entity_flags,
            result.mean_flagged_flags,
        )
        return build_report_messages(variables, instructions)

    async def report(
        self,
        result: NetworkResult,
        network_id: int,
        entity: str = "",
        instructions: str | None = None,
        callbacks: list[LLMCallback] | None = None,
    ) -> str:
        """Ask the LLM for the network (and, with ``entity``, exposure) report."""
        if self.chat is None:
            msg = "NetworkDetector has no chat function; pass one or use from_openai"
            raise ValueError(msg)
        messages = self.report_messages(result, network_id, entity, instructions)
        return await generate_report(messages, self.chat, callbacks)
