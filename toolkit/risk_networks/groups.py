# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl

from toolkit.helpers.constants import ATTRIBUTE_VALUE_SEPARATOR


def build_groups(
    value_cols: list[str], df_groups: pl.DataFrame, entity_col: str
) -> tuple[list, list]:
    group_links = []
    groups = []
    for value_col in value_cols:
        df_groups["attribute_col"] = value_col

        link_list = (
            df_groups.select([entity_col, "attribute_col", value_col])
            .to_numpy()
            .tolist()
        )
        for link in link_list:
            groups.add(f"{link[1]}{ATTRIBUTE_VALUE_SEPARATOR}{link[2]}")
        group_links.append(link_list)

    return group_links, groups
