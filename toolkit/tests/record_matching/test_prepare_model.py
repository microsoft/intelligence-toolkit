# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import polars as pl
import pytest

from toolkit.record_matching.prepare_model import format_dataset


class TestFormatDataset:
    @pytest.fixture()
    def selected_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ID1": [10, 20, 30],
                "Name": ["A", "B", "C"],
                "Attribute1": ["A1", "B1", "C1"],
                "Attribute6": ["DD2", "E22", "EF2"],
                "Attribute3": ["D2", "E2", "F2"],
            }
        )

    def test_empty(self) -> None:
        df_empty = pl.DataFrame()
        result = format_dataset(df_empty, "", "")
        assert result.is_empty()

    def test_add_with_id(self, selected_df) -> None:
        id = "ID1"
        name = "Name"
        entity_attribute_columns = ["Attribute1"]

        result = format_dataset(selected_df, entity_attribute_columns, name, id)

        assert "Entity ID" in result.columns
        assert "Entity name" in result.columns
        assert result["Entity ID"].to_list() == ["10", "20", "30"]
        assert len(result.columns) == 3

    def test_add_no_id(self, selected_df) -> None:
        name = "Name"
        entity_attribute_columns = ["Attribute1"]
        result = format_dataset(selected_df, entity_attribute_columns, name)

        assert "Attribute1" in result.columns
        assert "Entity ID" in result.columns
        assert "Entity name" in result.columns
        assert result["Entity ID"].to_list() == ["0", "1", "2"]
        assert len(result.columns) == 3

    def test_add_attributes_ordered(self, selected_df) -> None:
        name = "Name"
        entity_attribute_columns = ["Attribute3", "Attribute1"]
        result = format_dataset(selected_df, entity_attribute_columns, name)

        assert "Attribute1" in result.columns
        assert "Entity ID" in result.columns
        assert "Entity name" in result.columns

        assert result.columns == [
            "Entity ID",
            "Entity name",
            "Attribute1",
            "Attribute3",
        ]

    def test_add_attributes_empty(self, selected_df) -> None:
        name = "Name"
        entity_attribute_columns = []
        result = format_dataset(selected_df, entity_attribute_columns, name)

        assert "Entity ID" in result.columns
        assert "Entity name" in result.columns

        assert len(result.columns) == 2

    def test_add_attributes_no_max_rows(self, selected_df) -> None:
        name = "Name"
        entity_attribute_columns = []
        result = format_dataset(selected_df, entity_attribute_columns, name)

        assert result.height == 3

    def test_add_attributes_max_rows(self, selected_df) -> None:
        name = "Name"
        entity_attribute_columns = []
        max_rows = 2
        result = format_dataset(
            selected_df, entity_attribute_columns, name, max_rows=max_rows
        )

        assert result.height == 2
