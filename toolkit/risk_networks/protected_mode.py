# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import re
from typing import Any

numeric_pattern = r"^\d+(\.\d+)?$"


def is_numeric_column(column):
    return all(re.match(numeric_pattern, str(value)) for value in column)


def protect_data(
    data_df, value_cols, entity_col: str, entities_renamed=None, attributes_renamed=None
) -> tuple[Any, list, list[tuple]]:
    if entities_renamed is None:
        entities_renamed = []
    if attributes_renamed is None:
        attributes_renamed = []

    for value_col in value_cols:
        unique_names = data_df[entity_col].unique()
        for i, name in enumerate(unique_names, start=1):
            original_name = name
            new_name = f"Protected_Entity_{i}"
            name_exists = [x for x in entities_renamed if x[0] == name]
            if len(name_exists) == 0:
                entities_renamed.append(
                    (
                        original_name,
                        new_name,
                    )
                )
            else:
                new_name = name_exists[0][1]

            data_df = data_df.with_columns(
                [
                    data_df[entity_col]
                    .map_elements(
                        lambda x, new_name=new_name, name=name: new_name
                        if x == name
                        else x
                    )
                    .alias(entity_col)
                ]
            )

        unique_names_value = data_df[value_col].unique()
        is_numeric = is_numeric_column(data_df[value_col])
        if not is_numeric:
            for i, name in enumerate(unique_names_value, start=1):
                new_name = f"{value_col}_{i!s}"
                name_exists = [x for x in attributes_renamed if x[0] == name]
                name_exists_entity = [x for x in entities_renamed if x[0] == name]

                if len(name_exists) == 0 and len(name_exists_entity) == 0:
                    attributes_renamed.append(
                        (
                            name,
                            new_name,
                        )
                    )
                else:
                    if len(name_exists_entity) > 0:
                        new_name = name_exists_entity[0][1]
                    else:
                        new_name = name_exists[0][1]

                data_df = data_df.with_columns(
                    [
                        data_df[value_col]
                        .map_elements(
                            lambda x, new_name=new_name, name=name: new_name
                            if x == name
                            else x
                        )
                        .alias(value_col)
                    ]
                )
        else:
            attributes_renamed = [(name, name) for name in unique_names_value]

    return data_df, entities_renamed, attributes_renamed
