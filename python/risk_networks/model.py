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
    entity_links = []
    for value_col in columns_to_link:
        attribute_label = value_col
        if attribute_column_type == AttributeColumnType.CustomName:
            attribute_label = attribute_name

        if attribute_column_type in [
            AttributeColumnType.ColumnName,
            AttributeColumnType.CustomName,
        ]:
            data_df["attribute_col"] = attribute_label
            node_types.add(attribute_label)
            entity_links.append(
                data_df[[entity_id_column, "attribute_col", value_col]].to_numpy()
            )
        else:
            node_types.update(data_df[attribute_label].unique().tolist())
            entity_links.append(
                data_df[[entity_id_column, value_col, value_col]].to_numpy()
            )

    return entity_links, node_types
