# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_networks.models import (
    AttributeTable,
    EntityRecord,
    FlagFormat,
    FlagTable,
    ModelSummary,
    Network,
    NetworkConfig,
    NetworkResult,
    empty_result,
)
from itk_networks_testing import E, sample_frame
from pydantic import ValidationError


def test_attribute_table_rejects_missing_columns():
    with pytest.raises(ValidationError, match="no columns \\['zip'\\]"):
        AttributeTable(frame=sample_frame(), entity_column="name", columns=["zip"])


def test_attribute_table_rejects_entity_column_as_attribute():
    with pytest.raises(ValidationError, match="cannot also be linked"):
        AttributeTable(frame=sample_frame(), entity_column="name", columns=["name"])


def test_attribute_table_rejects_reserved_label():
    frame = sample_frame().rename({"phone": "ENTITY"})
    with pytest.raises(ValidationError, match="reserved"):
        AttributeTable(frame=frame, entity_column="name", columns=["ENTITY"])


def test_attribute_table_requires_at_least_one_column():
    with pytest.raises(ValidationError):
        AttributeTable(frame=sample_frame(), entity_column="name", columns=[])


def test_flag_table_defaults_to_count_format():
    table = FlagTable(frame=sample_frame(), entity_column="name", columns=["safety"])
    assert table.format is FlagFormat.Count


def test_config_defaults_match_legacy():
    config = NetworkConfig()
    assert config.max_attribute_degree == 10
    assert config.max_network_entities == 20
    assert config.supporting_attribute_types == ()
    assert config.removed_attributes == ()
    assert config.leiden_seed == 42


def test_config_rejects_zero_network_size():
    with pytest.raises(ValidationError):
        NetworkConfig(max_network_entities=0)


def test_model_summary_str_matches_legacy_text():
    summary = ModelSummary(entities=2, attributes=3, flags=4, groups=5, links=6)
    assert str(summary) == (
        "Number of entities: 2, Number of attributes: 3, Number of flags: 4, "
        "Number of groups: 5, Number of links: 6"
    )


def test_entity_record_row_order_matches_frame_columns():
    record = EntityRecord("A", 1, 0, 2, 3, 1, 1.5, 1.0)
    assert record.as_row() == ("A", 1, 0, 2, 3, 1, 1.5, 1.0)


def test_empty_result_has_typed_empty_frames():
    result = empty_result()
    assert result.graph.number_of_nodes() == 0
    assert result.flags.schema["count"] == pl.Int64
    assert result.summary == ModelSummary(0, 0, 0, 0, 0)
    assert result.records_summary == ""
    assert result.entities_frame.is_empty()
    assert result.inferred_links_frame.is_empty()


def test_result_network_lookup_and_sizes():
    result = empty_result()
    result.networks = [Network(0, (E("A"), E("B"))), Network(1, (E("C"),))]
    assert result.network(0).size == 2
    assert result.network_sizes == [2]
    assert result.records_summary == (
        "Networks identified: 2 (1 with multiple entities, maximum 2)"
    )
    with pytest.raises(ValueError, match="No network 5"):
        result.network(5)


def test_result_node_types_exclude_entities():
    result = NetworkResult.__new__(NetworkResult)  # type: ignore[call-arg]
    result = empty_result()
    result.graph.add_node(E("A"), type="ENTITY")
    result.graph.add_node("phone==1", type="phone")
    result.graph.add_node("email==x", type="email")
    assert result.node_types == ("email", "phone")
    assert result.entity_types == ["ENTITY", "email", "phone"]
    assert result.attributes == ["phone==1", "email==x"]
