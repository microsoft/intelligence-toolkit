# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import networkx as nx
import polars as pl

from toolkit.AI import utils
from toolkit.AI.client import OpenAIClient
from toolkit.detect_entity_networks import prompts
from toolkit.detect_entity_networks.classes import FlagAggregatorType
from toolkit.detect_entity_networks.config import (
    DEFAULT_MAX_ATTRIBUTE_DEGREE,
    ENTITY_LABEL,
)
from toolkit.detect_entity_networks.explore_networks import (
    build_network_from_entities,
    simplify_entities_graph,
)
from toolkit.detect_entity_networks.exposure_report import build_exposure_report
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
        self.inferred_links = {}

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

    def clear_inferred_nodes(self) -> None:
        self.inferred_links = {}

    def inferred_nodes_df(self) -> pl.DataFrame:
        link_list = [
            (text, n)
            for text, near in self.inferred_links.items()
            for n in near
            if text < n
        ]
        inferred_df = pl.DataFrame(link_list, schema=["text", "similar"])
        return inferred_df.with_columns(
            [
                pl.col("text").str.replace(
                    ENTITY_LABEL + ATTRIBUTE_VALUE_SEPARATOR, ""
                ),
                pl.col("similar").str.replace(
                    ENTITY_LABEL + ATTRIBUTE_VALUE_SEPARATOR, ""
                ),
            ]
        ).sort(["text", "similar"])

    def identify(
        self,
        max_network_entities: int | None = 20,
        max_attribute_degree: int | None = DEFAULT_MAX_ATTRIBUTE_DEGREE,
        supporting_attribute_types: list[str] | None = None,
    ) -> None:
        (trimmed_degrees, trimmed_nodes) = trim_nodeset(
            self.graph, max_attribute_degree, self.additional_trimmed_attributes
        )

        self.trimmed_attributes = pl.DataFrame(
            list(trimmed_degrees),
            schema=["Attribute", "Linked Entities"],
        ).sort("Linked Entities")

        (
            self.community_nodes,
            self.entity_to_community_ix,
        ) = build_networks(
            self.graph,
            trimmed_nodes,
            self.inferred_links,
            supporting_attribute_types,
            max_network_entities,
        )

        self.entity_records = build_entity_records(
            self.community_nodes,
            self.integrated_flags,
            self.inferred_links,
        )

    def community_sizes(self) -> list[int]:
        return [len(comm) for comm in self.community_nodes if len(comm) > 1]

    def get_records_summary(self) -> str:
        if len(self.community_nodes) > 0:
            comm_sizes = self.community_sizes()
            max_comm_size = max(comm_sizes)

            return f"Networks identified: {len(self.community_nodes)} ({len(comm_sizes)} with multiple entities, maximum {max_comm_size})"
        return ""

    def get_entity_df(self) -> pl.DataFrame:
        entity_df = pl.DataFrame(
            self.entity_records,
            schema=[
                "entity_id",
                "entity_flags",
                "network_id",
                "network_entities",
                "network_flags",
                "flagged",
                "flags/entity",
                "flagged/unflagged",
            ],
        )
        return entity_df.sort(by=["flagged/unflagged"], descending=True)

    def get_grouped_df(self) -> pl.DataFrame:
        show_df = self.get_entity_df()

        for group_links in self.group_links:
            selected_df = pl.DataFrame(
                group_links, schema=["entity_id", "group", "value"]
            )

            selected_df = selected_df.filter(pl.col("value").is_not_null())

            selected_df = selected_df.pivot(
                values="value",
                index="entity_id",
                columns="group",
                aggregate_function="first",
            )

            show_df = show_df.join(selected_df, on="entity_id", how="left")
        return show_df

    def get_exposure_report(
        self, selected_entity: str, selected_network: int | None = None
    ) -> pl.DataFrame:
        c_nodes = self.community_nodes[selected_network]

        return build_exposure_report(
            self.integrated_flags,
            selected_entity,
            c_nodes,
            self.graph,
            self.inferred_links,
        )

    def _get_entities_graph(self, selected_network: int) -> nx.Graph:
        c_nodes = self.community_nodes[selected_network]
        return build_network_from_entities(
            self.graph,
            self.entity_to_community_ix,
            self.integrated_flags,
            self.trimmed_attributes,
            self.inferred_links,
            c_nodes,
        )

    def get_merged_graph_df(self, selected_network: int) -> pl.DataFrame:
        simplified_graph = simplify_entities_graph(
            self._get_entities_graph(selected_network),
        )
        nodes = pl.DataFrame(
            [(n, d["type"], d["flags"]) for n, d in simplified_graph.nodes(data=True)],
            schema=["node", "type", "flags"],
        )
        links = pl.DataFrame(
            list(simplified_graph.edges()),
            schema=["source", "target"],
        )
        links = links.with_columns(
            pl.col("target")
            .apply(lambda x: x.split(ATTRIBUTE_VALUE_SEPARATOR)[0])
            .alias("attribute")
        )

        return nodes, links

    def generate_report(
        self,
        selected_network,
        selected_entity: str | None = "",
        ai_instructions: str | None = prompts.user_prompt,
    ):
        nodes_merged, links_merged = self.get_merged_graph_df(selected_network)
        variables = {
            "entity_id": selected_entity,
            "network_id": selected_network,
            "max_flags": self.max_entity_flags,
            "mean_flags": self.mean_flagged_flags,
            "exposure": "",
            "network_nodes": nodes_merged.write_csv(),
            "network_edges": links_merged.write_csv(),
        }
        messages = utils.generate_messages(
            ai_instructions,
            prompts.list_prompts["report_prompt"],
            variables,
            prompts.list_prompts["safety_prompt"],
        )
        return OpenAIClient(self.ai_configuration).generate_chat(messages, stream=False)