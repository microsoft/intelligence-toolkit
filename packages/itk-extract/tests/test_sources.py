# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

import json

import polars as pl
import pytest
from itk_extract.models import Source
from itk_extract.sources import check_unique_ids, from_frame, from_texts


def test_from_texts_numbers_ids() -> None:
    sources = from_texts(["a", "b"], prefix="doc")
    assert [s.id for s in sources] == ["doc-0", "doc-1"]
    assert [s.text for s in sources] == ["a", "b"]


def test_from_frame_renders_rows_as_json() -> None:
    frame = pl.DataFrame({"id": [10, 11], "name": ["Ann", "Ben"], "age": [30, None]})
    sources = from_frame(frame, id_column="id")
    assert [s.id for s in sources] == ["10", "11"]
    assert json.loads(sources[0].text) == {"id": 10, "name": "Ann", "age": 30}
    assert json.loads(sources[1].text)["age"] is None
    assert sources[1].meta == {"id": 11, "name": "Ben", "age": None}


def test_from_frame_restricts_columns_and_defaults_ids() -> None:
    frame = pl.DataFrame({"name": ["Ann"], "secret": ["x"]})
    sources = from_frame(frame, columns=["name"])
    assert sources[0].id == "row-0"
    assert json.loads(sources[0].text) == {"name": "Ann"}
    assert sources[0].meta["secret"] == "x"


def test_from_frame_rejects_missing_columns() -> None:
    frame = pl.DataFrame({"name": ["Ann"]})
    with pytest.raises(ValueError, match="no columns"):
        from_frame(frame, columns=["nope"])
    with pytest.raises(ValueError, match="no id column"):
        from_frame(frame, id_column="nope")


def test_from_frame_rejects_duplicate_ids() -> None:
    frame = pl.DataFrame({"id": [1, 1]})
    with pytest.raises(ValueError, match="Duplicate source id"):
        from_frame(frame, id_column="id")


def test_check_unique_ids() -> None:
    check_unique_ids([Source(id="a", text=""), Source(id="b", text="")])
    with pytest.raises(ValueError, match="Duplicate"):
        check_unique_ids([Source(id="a", text=""), Source(id="a", text="")])


def test_source_requires_non_empty_id() -> None:
    with pytest.raises(ValueError, match="id"):
        Source(id="", text="x")
