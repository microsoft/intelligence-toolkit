# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from python.risk_networks.config import AttributeColumnType


def prepare_entity_attribute(
    data_df,
    entity_id_column: str,
    attribute_column_type: AttributeColumnType,
    columns_to_link: list[str],
    attribute_name=None,
) -> tuple[list, set]:
    node_types = set()
    attribute_links = []
    for value_col in columns_to_link:
        attribute_label = value_col
        if attribute_column_type == AttributeColumnType.CustomName.value:
            attribute_label = attribute_name

        data_df["attribute_col"] = attribute_label
        node_types.add(attribute_label)
        attribute_links.append(
            data_df[[entity_id_column, "attribute_col", value_col]].to_numpy()
        )
    return attribute_links, node_types
