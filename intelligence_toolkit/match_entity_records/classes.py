# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from typing import TypedDict

import polars as pl
from pydantic import BaseModel


class RecordsModel(BaseModel):
    dataframe: pl.DataFrame
    name_column: str
    columns: list[str]
    dataframe_name: str | None = None
    id_column: str | None = None

    class Config:
        arbitrary_types_allowed = True


class AttributeToMatch(TypedDict):
    label: str | None = None
    columns: list[str]