# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


def prepare_entity_attribute(
    data_df,
    entity_id_column: str,
    columns_to_link: list[str],
) -> list:
    attribute_links = []
    for value_col in columns_to_link:
        data_df["attribute_col"] = value_col
        attribute_links.append(
            data_df[[entity_id_column, "attribute_col", value_col]].to_numpy()
        )
    return attribute_links
