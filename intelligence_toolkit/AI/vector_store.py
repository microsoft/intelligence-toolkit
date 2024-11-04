# # Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# # Licensed under the MIT license. See LICENSE file in the project.
# #
from typing import Any

import duckdb
import lancedb
import pyarrow as pa
from pandas import DataFrame

from intelligence_toolkit.helpers.constants import CACHE_PATH

table_missing_msg = "Table not initialized"


class VectorStore:
    table = None
    duckdb_data = None

    def __init__(
        self,
        table_name: str | None = None,
        path: str = CACHE_PATH,
        schema: pa.Schema = None,
    ):
        self.db_connection = lancedb.connect(path)
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
        if isinstance(texts, str):
            texts = [texts]
        arrow_data = self.duckdb_data
        query = f"SELECT DISTINCT * FROM arrow_data WHERE {column} IN {tuple(texts)}"
        return duckdb.execute(query).df()

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

    def drop_db(self):
        self.db_connection.drop_database()
