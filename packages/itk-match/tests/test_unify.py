# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_match.models import Dataset
from itk_match.unify import attribute_columns, unify


def test_unify_aligns_columns_and_builds_record_ids(dataset_one, dataset_two):
    records = unify([dataset_one, dataset_two])

    assert records.columns == [
        "record_id",
        "dataset",
        "entity_id",
        "name",
        "address",
        "city",
        "phone",
    ]
    assert all(dtype == pl.Utf8 for dtype in records.dtypes)
    assert records.get_column("record_id").to_list() == [
        "D1:1",
        "D1:2",
        "D1:3",
        "D2:0",
        "D2:1",
        "D2:2",
    ]


def test_unify_drops_rows_with_empty_names(dataset_two):
    records = unify([dataset_two])
    assert records.height == 3
    assert "" not in records.get_column("name").to_list()


def test_unify_uses_row_index_when_no_id_column(dataset_two):
    records = unify([dataset_two])
    assert records.get_column("entity_id").to_list() == ["0", "1", "2"]


def test_unify_fills_missing_labels_with_null(dataset_one, dataset_two):
    records = unify([dataset_one, dataset_two])
    phones = records.filter(pl.col("dataset") == "D1").get_column("phone")
    assert phones.null_count() == 3


def test_unify_caps_rows_per_dataset(dataset_one, dataset_two):
    records = unify([dataset_one, dataset_two], max_rows_per_dataset=1)
    assert records.get_column("dataset").to_list() == ["D1", "D2"]


def test_unify_rejects_duplicate_dataset_names(dataset_one):
    with pytest.raises(ValueError, match="unique"):
        unify([dataset_one, dataset_one])


def test_unify_of_nothing_is_an_empty_frame_with_reserved_columns():
    records = unify([])
    assert records.height == 0
    assert records.columns == ["record_id", "dataset", "entity_id", "name"]


def test_attribute_columns_excludes_reserved(dataset_one):
    assert attribute_columns(unify([dataset_one])) == ["address", "city"]


def test_unify_strips_whitespace():
    ds = Dataset(
        name="D",
        frame=pl.DataFrame({"n": ["  Acme  "], "a": [" 1 Main St "]}),
        name_column="n",
        attributes={"a": "address"},
    )
    row = unify([ds]).row(0, named=True)
    assert row["name"] == "Acme"
    assert row["address"] == "1 Main St"
