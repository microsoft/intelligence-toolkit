# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
from itk_networks.models import Network
from itk_networks.prepare import flag_links, group_links, integrate_flags
from itk_networks.records import (
    entities_frame,
    entity_records,
    flag_totals,
    grouped_frame,
    network_flags,
)
from itk_networks_testing import E, flag_table, group_table


def _flags() -> pl.DataFrame:
    return integrate_flags(flag_links(flag_table()))[0]


def test_flag_totals_per_qualified_entity():
    totals = flag_totals(_flags())
    assert totals[E("A")] == 2
    assert totals[E("B")] == 0
    assert flag_totals(_flags().clear()) == {}


def test_network_flags_basic_statistics():
    totals = flag_totals(_flags())
    assert network_flags((E("A"), E("B")), totals, {}) == (2, 1, 1.0, 1.0, 2)
    assert network_flags((E("C"), E("D")), totals, {}) == (1, 1, 1.0, 0.5, 2)
    assert network_flags((E("E"),), totals, {}) == (0, 0, 0.0, 0.0, 1)


def test_network_flags_without_any_flags_reports_zeros():
    assert network_flags((E("A"), E("B")), {}, {}) == (0, 0, 0.0, 0.0, 2)


def test_network_flags_count_inferred_neighbours_once():
    totals = flag_totals(_flags())
    links = {
        E("A"): frozenset({E("C")}),
        E("B"): frozenset({E("C")}),
        E("C"): frozenset({E("A"), E("B")}),
    }
    total, flagged, per_unflagged, per_entity, size = network_flags(
        (E("A"), E("B")), totals, links
    )
    assert (total, flagged, size) == (3, 2, 3)
    assert per_unflagged == 2.0
    assert per_entity == 1.0


def test_network_flags_all_flagged_gives_zero_ratio():
    totals = {E("A"): 1, E("C"): 1}
    assert network_flags((E("A"), E("C")), totals, {})[2] == 0.0


def test_entity_records_rows_in_network_order():
    networks = [Network(0, (E("A"), E("B"))), Network(1, (E("C"), E("D")))]
    records = entity_records(networks, _flags())
    assert [r.as_row() for r in records] == [
        ("A", 2, 0, 2, 2, 1, 1.0, 1.0),
        ("B", 0, 0, 2, 2, 1, 1.0, 1.0),
        ("C", 1, 1, 2, 1, 1, 0.5, 1.0),
        ("D", 0, 1, 2, 1, 1, 0.5, 1.0),
    ]


def test_entities_frame_columns_and_sort():
    networks = [Network(0, (E("E"),)), Network(1, (E("A"), E("B")))]
    frame = entities_frame(entity_records(networks, _flags()))
    assert frame.columns == [
        "entity_id",
        "entity_flags",
        "network_id",
        "network_entities",
        "network_flags",
        "flagged",
        "flags/entity",
        "flagged/unflagged",
    ]
    assert frame.get_column("entity_id").to_list() == ["A", "B", "E"]
    assert frame.schema["flags/entity"] == pl.Float64


def test_entities_frame_empty_keeps_schema():
    frame = entities_frame([])
    assert frame.is_empty()
    assert frame.schema["network_id"] == pl.Int64


def test_grouped_frame_adds_one_column_per_group():
    networks = [Network(0, (E("A"), E("B"))), Network(1, (E("F"),))]
    entities = entities_frame(entity_records(networks, _flags()))
    grouped = grouped_frame(entities, group_links(group_table()))
    assert grouped.columns[-1] == "sector"
    by_entity = dict(grouped.select("entity_id", "sector").rows())
    assert by_entity == {"A": "retail", "B": "retail", "F": None}


def test_grouped_frame_takes_first_value_and_ignores_empty_groups():
    entities = entities_frame(entity_records([Network(0, (E("A"),))], _flags()))
    links = pl.DataFrame({
        "entity": ["A", "A"],
        "group": ["sector", "sector"],
        "value": ["x", "y"],
    })
    assert grouped_frame(entities, links).get_column("sector").to_list() == ["x"]
    assert grouped_frame(entities, links.clear()).columns == entities.columns
