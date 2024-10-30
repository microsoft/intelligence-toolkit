# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl

from toolkit.detect_entity_networks.classes import FlagAggregatorType
from toolkit.detect_entity_networks.config import ENTITY_LABEL
from toolkit.detect_entity_networks.identify_networks import (
    build_entity_records,
    build_networks,
    trim_nodeset,
)
from toolkit.detect_entity_networks.index_and_infer import index_nodes, infer_nodes
from toolkit.detect_entity_networks.prepare_model import (
    build_flag_links,
    build_flags,
    build_groups,
    build_main_graph,
    generate_attribute_links,
)
from toolkit.helpers.classes import IntelligenceWorkflow
from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR


class DetectEntityNetworks(IntelligenceWorkflow):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attribute_links = []
        self.attributes_list = []
        self.flag_links = []
        self.group_links = []
        self.additional_trimmed_attributes = []
        self.graph = None
        self.integrated_flags = pl.DataFrame()
        self.node_types = set()

    def get_fuzzy_options(self) -> list[str]:
        return sorted(
            [
                ENTITY_LABEL,
                *list(self.node_types),
            ]
        )

    def get_attributes(self) -> pl.DataFrame:
        return pl.DataFrame(self.attributes_list, columns=["Attribute"])

    def _build_graph(self) -> nx.Graph:
        self.graph = build_main_graph(self.attribute_links)

    def remove_attributes(self, selected_rows: pl.DataFrame) -> list[str]:
        self.additional_trimmed_attributes = selected_rows["Attribute"].tolist()

    def add_attribute_links(
        self, data_df: pl.DataFrame, entity_id_column: str, columns_to_link: list[str]
    ):
        links = generate_attribute_links(data_df, entity_id_column, columns_to_link)
        self.attribute_links.extend(links)
        for attribute_link in links:
            self.node_types.add(attribute_link[0][1])
        self._build_graph()
        return self.graph

    def add_flag_links(
        self,
        data_df: pl.DataFrame,
        entity_id_column: str,
        flag_column: str,
        flag_format: FlagAggregatorType,
    ) -> None:
        links = build_flag_links(
            data_df,
            entity_id_column,
            flag_format,
            flag_column,
        )
        self.flag_links.extend(links)

        (
            self.integrated_flags,
            self.max_entity_flags,
            self.mean_flagged_flags,
        ) = build_flags(self.flag_links)

    def add_groups(
        self,
        data_df: pl.DataFrame,
        entity_id_column: str,
        value_cols: list[str],
    ) -> None:
        self.group_links = build_groups(
            value_cols,
            data_df,
            entity_id_column,
        )

    def get_model_summary_data(self) -> str:
        num_entities = 0
        num_attributes = 0
        num_flags = 0
        groups = set()
        for link_list in self.group_links:
            for link in link_list:
                groups.add(f"{link[1]}{ATTRIBUTE_VALUE_SEPARATOR}{link[2]}")

        if self.graph is not None:
            all_nodes = self.graph.nodes()
            entity_nodes = [node for node in all_nodes if node.startswith(ENTITY_LABEL)]
            self.attributes_list = [
                node for node in all_nodes if not node.startswith(ENTITY_LABEL)
            ]
            num_entities = len(entity_nodes)
            num_attributes = len(all_nodes) - num_entities

        if len(self.integrated_flags) > 0:
            num_flags = self.integrated_flags["count"].sum()

        return {
            "entities": num_entities,
            "attributes": num_attributes,
            "flags": num_flags,
            "groups": len(groups),
            "links": len(self.graph.edges()),
        }

    def get_model_summary_value(self):
        summary = self.get_model_summary_data()
        return f"Number of entities: {summary['entities']}, Number of attributes: {summary['attributes']}, Number of flags: {summary['flags']}, Number of groups: {summary['groups']}, Number of links: {summary['links']}"

    async def index_nodes(self, node_types: list[str]) -> None:
        (
            self.embedded_texts,
            self.nearest_text_distances,
            self.nearest_text_indices,
        ) = await index_nodes(
            node_types,
            self.graph,
        )

    async def infer_nodes(self, similarity_threshold: float) -> None:
        self.inferred_links = infer_nodes(
            similarity_threshold,
            self.embedded_texts,
            self.nearest_text_indices,
            self.nearest_text_distances,
        )

    def identify(
        self,
        max_network_entities: int | None = None,
        max_attribute_degree: int | None = None,
        supporting_attribute_types: list[str] | None = None,
    ) -> None:
        (trimmed_degrees, trimmed_nodes) = trim_nodeset(
            self.graph, self.additional_trimmed_attributes, max_attribute_degree
        )

        self.trimmed_attributes.value = pl.DataFrame(
            trimmed_degrees,
            schema=["Attribute", "Linked Entities"],
        ).sort("Linked Entities")

        (
            community_nodes,
            self.entity_to_community_ix,
        ) = build_networks(
            self.graph,
            trimmed_nodes,
            self.inferred_links,
            supporting_attribute_types,
            max_network_entities,
        )

        self.entity_records = build_entity_records(
            community_nodes,
            self.integrated_flags,
            self.inferred_links,
        )
