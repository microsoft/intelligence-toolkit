# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
from itk_networks.models import AttributeTable, FlagFormat, FlagTable, ModelSummary
from itk_networks.prepare import (
    attribute_links,
    build_graph,
    flag_links,
    group_links,
    integrate_flags,
    model_summary,
)
from itk_networks_testing import (
    A,
    E,
    attribute_table,
    flag_table,
    group_table,
    sample_frame,
)


def test_attribute_links_are_long_cleaned_and_drop_nulls():
    links = attribute_links(attribute_table())
    assert links.columns == ["entity", "attribute", "value"]
    assert links.height == 6 + 5 + 6  # phone, email (one null), city
    assert links.filter(pl.col("entity") == "F").get_column("attribute").to_list() == [
        "phone",
        "city",
    ]


def test_attribute_links_clean_values_and_entities():
    frame = pl.DataFrame({"id": ["Acme, Inc."], "mail": ["a.b@x.com"]})
    table = AttributeTable(frame=frame, entity_column="id", columns=["mail"])
    assert attribute_links(table).row(0) == ("Acme Inc", "mail", "ab@xcom")


def test_attribute_links_accept_integer_columns():
    frame = pl.DataFrame({"id": [1, 2], "phone": [555, None]})
    table = AttributeTable(frame=frame, entity_column="id", columns=["phone"])
    assert attribute_links(table).rows() == [("1", "phone", "555")]


def test_build_graph_nodes_edges_and_types():
    graph = build_graph(attribute_links(attribute_table()))
    assert graph.nodes[E("A")]["type"] == "ENTITY"
    assert graph.nodes[A("phone", "111")]["type"] == "phone"
    assert graph.has_edge(E("A"), A("phone", "111"))
    assert graph.has_edge(E("B"), A("phone", "111"))
    assert graph.edges[E("A"), A("phone", "111")]["type"] == "phone"
    assert graph.degree(A("city", "Metro")) == 6


def test_build_graph_adds_equality_edges_between_types_sharing_a_value():
    frame = pl.DataFrame({
        "id": ["A", "B"],
        "owner": ["Paris", "x"],
        "city": ["y", "Paris"],
    })
    table = AttributeTable(frame=frame, entity_column="id", columns=["owner", "city"])
    graph = build_graph(attribute_links(table))
    assert graph.has_edge(A("city", "Paris"), A("owner", "Paris"))
    assert graph.edges[A("city", "Paris"), A("owner", "Paris")]["type"] == "equality"


def test_build_graph_from_empty_links_is_empty():
    graph = build_graph(attribute_links(attribute_table()).clear())
    assert graph.number_of_nodes() == 0


def test_flag_links_count_format_sums_per_entity():
    frame = pl.DataFrame({
        "name": ["A", "A", "B"],
        "safety": [1, 2, 0],
        "pay": [0, 1, 0],
    })
    links = flag_links(flag_table(frame))
    assert links.columns == ["entity", "type", "flag", "count"]
    by_key = {(r[0], r[1]): r[3] for r in links.rows()}
    assert by_key["A", "safety"] == 3
    assert by_key["A", "pay"] == 1
    assert by_key["B", "safety"] == 0
    assert links.filter(pl.col("type") == "safety").get_column("flag").to_list() == [
        "safety",
        "safety",
    ]


def test_flag_links_instance_format_names_flag_after_sum_and_counts_one():
    frame = pl.DataFrame({"name": ["A", "A"], "safety": [1, 2], "pay": [0, 0]})
    links = flag_links(flag_table(frame, FlagFormat.Instance))
    assert links.rows() == [("A", "safety", "3", 1), ("A", "pay", "0", 1)]


def test_flag_links_read_booleans_and_text_booleans():
    frame = pl.DataFrame({
        "name": ["A", "B", "C"],
        "safety": ["true", " False ", None],
        "pay": [True, False, True],
    })
    links = flag_links(flag_table(frame))
    counts = {(r[0], r[1]): r[3] for r in links.rows()}
    assert counts["A", "safety"] == 1
    assert counts["B", "safety"] == 0
    assert counts["C", "safety"] == 0
    assert counts["C", "pay"] == 1


def test_flag_links_drop_null_entities_and_clean_names():
    frame = pl.DataFrame({"name": ["A.", None], "safety": [1, 1], "pay": [0, 0]})
    links = flag_links(flag_table(frame))
    assert links.get_column("entity").unique().to_list() == ["A"]


def test_integrate_flags_sums_and_reports_max_and_mean():
    flags, max_flags, mean_flags = integrate_flags(flag_links(flag_table()))
    assert flags.columns == ["entity", "type", "flag", "count", "qualified_entity"]
    totals = dict(flags.group_by("qualified_entity").agg(pl.col("count").sum()).rows())
    assert totals[E("A")] == 2
    assert totals[E("C")] == 1
    assert totals[E("F")] == 1
    assert totals[E("B")] == 0
    assert max_flags == 2
    assert mean_flags == round((2 + 1 + 1) / 3, 2)


def test_integrate_flags_merges_duplicate_keys_across_tables():
    links = pl.concat([flag_links(flag_table()), flag_links(flag_table())])
    flags, max_flags, _ = integrate_flags(links)
    row = flags.filter((pl.col("entity") == "A") & (pl.col("type") == "safety"))
    assert row.get_column("count").to_list() == [4]
    assert max_flags == 4


def test_integrate_flags_empty():
    flags, max_flags, mean_flags = integrate_flags(flag_links(flag_table()).clear())
    assert flags.is_empty()
    assert (max_flags, mean_flags) == (0, 0.0)


def test_group_links_drop_nulls():
    links = group_links(group_table())
    assert links.columns == ["entity", "group", "value"]
    assert links.height == 5
    assert "F" not in links.get_column("entity").to_list()


def test_model_summary_counts():
    graph = build_graph(attribute_links(attribute_table()))
    flags, _, _ = integrate_flags(flag_links(flag_table()))
    summary = model_summary(graph, flags, group_links(group_table()))
    # 6 entities; attributes: 4 phones, 4 emails, 1 city
    assert summary == ModelSummary(
        entities=6, attributes=9, flags=4, groups=2, links=17
    )


def test_flag_table_with_single_column():
    table = FlagTable(frame=sample_frame(), entity_column="name", columns=["safety"])
    assert flag_links(table).get_column("type").unique().to_list() == ["safety"]
