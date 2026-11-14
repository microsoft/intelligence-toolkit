# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_match.models import Dataset, Group, MatchConfig, Pair
from pydantic import ValidationError


def test_dataset_rejects_missing_columns():
    with pytest.raises(ValidationError, match="no columns \\['nope'\\]"):
        Dataset(name="D", frame=pl.DataFrame({"n": ["a"]}), name_column="nope")


def test_dataset_rejects_reserved_labels():
    frame = pl.DataFrame({"n": ["a"], "x": ["b"]})
    with pytest.raises(ValidationError, match="reserved"):
        Dataset(name="D", frame=frame, name_column="n", attributes={"x": "record_id"})


def test_dataset_rejects_name_column_as_attribute():
    frame = pl.DataFrame({"n": ["a"]})
    with pytest.raises(ValidationError, match="cannot also be an attribute"):
        Dataset(name="D", frame=frame, name_column="n", attributes={"n": "label"})


def test_dataset_labels_are_the_canonical_names():
    frame = pl.DataFrame({"n": ["a"], "x": ["b"], "y": ["c"]})
    ds = Dataset(
        name="D", frame=frame, name_column="n", attributes={"x": "addr", "y": "addr"}
    )
    assert ds.labels == {"addr"}


def test_match_config_bounds_thresholds():
    with pytest.raises(ValidationError):
        MatchConfig(max_distance=-0.1)
    with pytest.raises(ValidationError):
        MatchConfig(min_name_similarity=1.5)
    with pytest.raises(ValidationError):
        MatchConfig(name_scorer="levenshtein")  # type: ignore[arg-type]


def test_match_config_defaults_are_frozen():
    config = MatchConfig()
    assert config.min_name_similarity == pytest.approx(0.75)
    assert config.neighbors == 50
    with pytest.raises(ValidationError):
        config.max_distance = 1.0  # type: ignore[misc]


def test_match_config_allows_zero_name_similarity() -> None:
    """Keep the name threshold explicitly disableable."""
    assert MatchConfig(min_name_similarity=0.0).min_name_similarity == pytest.approx(
        0.0
    )


def test_pair_requires_ordered_rows():
    assert Pair(0, 1, 0.0, 1.0).left == 0
    with pytest.raises(ValueError, match="left < right"):
        Pair(1, 1, 0.0, 1.0)


def test_group_size():
    assert Group(id=0, rows=(0, 4, 7), name_similarity=0.5).size == 3
