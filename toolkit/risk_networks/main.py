# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl

from toolkit.risk_networks.config import ENTITY_LABEL
from toolkit.risk_networks.explore_networks import (
    build_network_from_entities,
    get_entity_graph,
)
from toolkit.risk_networks.identify_networks import (
    build_entity_records,
    build_networks,
    trim_nodeset,
)
from toolkit.risk_networks.prepare_model import build_model_with_attributes


def build_simple_network_data(
    input_dataframe, entity_id_column, columns_to_link, selected_network_id=0
):
    main_graph = build_model_with_attributes(
        input_dataframe, entity_id_column, columns_to_link
    )
    (trimmed_degrees, trimmed_nodes) = trim_nodeset(
        main_graph,
    )

    (
        community_nodes,
        entity_to_community,
    ) = build_networks(
        main_graph,
        trimmed_nodes,
    )

    entity_records = build_entity_records(
        community_nodes,
    )

    selected_nodes = community_nodes[selected_network_id]
    network_entities_graph = build_network_from_entities(
        graph=main_graph,
        entity_to_community=entity_to_community,
        trimmed_attributes=trimmed_degrees,
        selected_nodes=selected_nodes,
    )

    entity_selected = ""

    columns_to_link.append(ENTITY_LABEL)
    nodes, edges = get_entity_graph(
        network_entities_graph,
        entity_selected,
        columns_to_link,
    )
    return entity_records, nodes, edges
