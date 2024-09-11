# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import polars as pl
import pytest

from toolkit.match_entity_records.prepare_model import (
    build_attribute_options,
    format_dataset,
)


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
        result = format_dataset(df_empty, [], "")
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


class TestBuildAttributeOptions:
    def test_empty(self) -> None:
        matching_dfs = {}
        result = build_attribute_options(matching_dfs)
        assert result == []

    def test_single(self) -> None:
        matching_dfs = {
            "dataset1": pl.DataFrame(
                {
                    "Entity ID": [1],
                    "Entity name": ["A"],
                    "Attribute1": ["A1"],
                    "Attribute2": ["A2"],
                }
            )
        }
        result = build_attribute_options(matching_dfs)
        assert result == [
            "Attribute1::dataset1",
            "Attribute2::dataset1",
        ]

    def test_multiple(self) -> None:
        matching_dfs = {
            "dataset1": pl.DataFrame(
                {
                    "Entity ID": [1],
                    "Entity name": ["A"],
                    "Attribute1": ["A1"],
                    "Attribute2": ["A2"],
                }
            ),
            "dataset2": pl.DataFrame(
                {
                    "Entity ID": [1],
                    "Entity name": ["A"],
                    "Attribute3": ["A3"],
                    "Attribute4": ["A4"],
                }
            ),
        }
        result = build_attribute_options(matching_dfs)
        assert result == [
            "Attribute1::dataset1",
            "Attribute2::dataset1",
            "Attribute3::dataset2",
            "Attribute4::dataset2",
        ]

    def test_order(self) -> None:
        matching_dfs = {
            "dataset2": pl.DataFrame(
                {
                    "Entity ID": [5],
                    "Entity name": ["A"],
                    "Attribute3": ["A3"],
                    "Attribute0": ["A4"],
                }
            ),
            "dataset1": pl.DataFrame(
                {
                    "Entity ID": [2, 3],
                    "Entity name": ["A", "B"],
                    "Attribute1": ["A1", "B1"],
                    "Attribute2": ["A2", "B2"],
                }
            ),
        }
        result = build_attribute_options(matching_dfs)
        assert result == [
            "Attribute0::dataset2",
            "Attribute1::dataset1",
            "Attribute2::dataset1",
            "Attribute3::dataset2",
        ]
        assert result == sorted(result)
