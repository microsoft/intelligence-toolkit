# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
from typing import Any

import duckdb
import lancedb
import pyarrow as pa
from pandas import DataFrame

from itk_core.paths import CACHE_PATH

table_missing_msg = "Table not initialized"


class VectorStore:
    table = None
    table_name: str | None = None
    duckdb_data = None

    def __init__(
        self,
        table_name: str | None = None,
        path: str = CACHE_PATH,
        schema: pa.Schema = None,
    ):
        if "://" not in str(path):
            os.makedirs(path, exist_ok=True)
        self.db_connection = lancedb.connect(path)
        self.table_name = table_name
        if table_name is not None:
            self.table = self.db_connection.create_table(
                table_name, schema=schema, exist_ok=True
            )
            self.duckdb_data = self.table.to_lance()

    def save(self, items: list[Any]) -> None:
        if self.table is None:
            raise ValueError(table_missing_msg)
        self.table.add(items)

    def search_by_column(self, texts: list[str] | str, column: str) -> DataFrame:
        if self.table is None:
            raise ValueError(table_missing_msg)
        if column not in self.table.schema.names:
            msg = f"Unknown column: {column}"
            raise ValueError(msg)
        if isinstance(texts, str):
            texts = [texts]
        with duckdb.connect() as con:
            con.register("arrow_data", self.duckdb_data)
            return con.execute(
                f'SELECT DISTINCT * FROM arrow_data WHERE "{column}" IN (SELECT UNNEST(?))',
                [list(texts)],
            ).df()

    def search_by_vector(self, vector: list[float], k: int = 10) -> list[dict]:
        if self.table is None:
            raise ValueError(table_missing_msg)

        return self.table.search(vector).limit(k).to_list()

    def update_duckdb_data(self) -> None:
        if self.table is None:
            raise ValueError(table_missing_msg)
        self.duckdb_data = self.table.to_lance()

    def drop_table(self) -> None:
        if self.table is None:
            raise ValueError(table_missing_msg)
        self.db_connection.drop_table(self.table_name)
        self.table = None
        self.duckdb_data = None

    def drop_db(self):
        self.db_connection.drop_database()
