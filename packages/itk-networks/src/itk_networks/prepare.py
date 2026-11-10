# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Turn entity tables into link frames, the attribute graph and the flags frame."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import polars as pl

from itk_networks.models import (
    ATTRIBUTE,
    ATTRIBUTE_LINK_SCHEMA,
    COUNT,
    ENTITY,
    FLAG,
    FLAG_LINK_SCHEMA,
    FLAGS_SCHEMA,
    GROUP,
    GROUP_LINK_SCHEMA,
    QUALIFIED_ENTITY,
    TYPE,
    VALUE,
    AttributeTable,
    FlagFormat,
    FlagTable,
    GroupTable,
    ModelSummary,
    cast_frame,
)
from itk_networks.text import (
    ENTITY_LABEL,
    EQUALITY_EDGE_TYPE,
    clean_text,
    is_entity,
    qualify,
    qualify_entity,
)


def _cleaned(column: str) -> pl.Expr:
    return pl.col(column).map_elements(clean_text, return_dtype=pl.Utf8)


def _links(
    frame: pl.DataFrame,
    entity_column: str,
    columns: list[str],
    middle: str,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Long frame ``entity, <middle>, value`` with one row per non-null cell."""
    parts = [
        frame.filter(pl.col(entity_column).is_not_null() & pl.col(col).is_not_null())
        .select(
            _cleaned(entity_column).alias(ENTITY),
            pl.lit(col).alias(middle),
            _cleaned(col).alias(VALUE),
        )
        .pipe(cast_frame, schema)
        for col in columns
    ]
    return pl.concat(parts) if parts else pl.DataFrame(schema=schema)


def attribute_links(table: AttributeTable) -> pl.DataFrame:
    """``entity, attribute, value`` rows, cleaned, null cells dropped."""
    return _links(
        table.frame,
        table.entity_column,
        table.columns,
        ATTRIBUTE,
        ATTRIBUTE_LINK_SCHEMA,
    )


def group_links(table: GroupTable) -> pl.DataFrame:
    """``entity, group, value`` rows, cleaned, null cells dropped."""
    return _links(
        table.frame, table.entity_column, table.columns, GROUP, GROUP_LINK_SCHEMA
    )


def _as_count(column: str) -> pl.Expr:
    """true/false become 1/0; anything else is cast to an integer or null."""
    normalized = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return (
        pl.when(pl.col(column).is_null())
        .then(None)
        .when(normalized == "true")
        .then(pl.lit(1))
        .when(normalized == "false")
        .then(pl.lit(0))
        .otherwise(pl.col(column))
        .cast(pl.Int64, strict=False)
    )


def flag_links(table: FlagTable) -> pl.DataFrame:
    """``entity, type, flag, count`` rows summed per entity and column.

    ``Count`` format names the flag after the column and keeps the sum;
    ``Instance`` format names the flag after the sum and counts one, exactly
    as the legacy ``build_flag_links`` did.
    """
    frame = table.frame.filter(pl.col(table.entity_column).is_not_null())
    summed = (
        frame.with_columns(_cleaned(table.entity_column).alias(ENTITY))
        .with_columns([_as_count(col).alias(col) for col in table.columns])
        .group_by(ENTITY, maintain_order=True)
        .agg([pl.col(col).sum() for col in table.columns])
    )
    parts = []
    for col in table.columns:
        if table.format == FlagFormat.Count:
            part = summed.select(
                pl.col(ENTITY),
                pl.lit(col).alias(TYPE),
                pl.lit(col).alias(FLAG),
                pl.col(col).alias(COUNT),
            )
        else:
            part = summed.select(
                pl.col(ENTITY),
                pl.lit(col).alias(TYPE),
                pl.col(col).cast(pl.Utf8).alias(FLAG),
                pl.lit(1).alias(COUNT),
            )
        parts.append(cast_frame(part, FLAG_LINK_SCHEMA))
    return pl.concat(parts) if parts else pl.DataFrame(schema=FLAG_LINK_SCHEMA)


def integrate_flags(links: pl.DataFrame) -> tuple[pl.DataFrame, int, float]:
    """Flags summed per ``entity, type, flag`` plus max and mean entity totals.

    The mean is over entities with at least one flag, rounded to two places.
    """
    if links.is_empty():
        return pl.DataFrame(schema=FLAGS_SCHEMA), 0, 0.0
    flags = (
        links.group_by([ENTITY, TYPE, FLAG], maintain_order=True)
        .agg(pl.col(COUNT).sum())
        .with_columns(
            pl.col(ENTITY)
            .map_elements(qualify_entity, return_dtype=pl.Utf8)
            .alias(QUALIFIED_ENTITY)
        )
        .pipe(cast_frame, FLAGS_SCHEMA)
    )
    totals = flags.group_by(QUALIFIED_ENTITY).agg(pl.col(COUNT).sum())
    max_flags = int(totals.get_column(COUNT).max() or 0)
    flagged = totals.filter(pl.col(COUNT) > 0)
    mean = flagged.get_column(COUNT).mean() if not flagged.is_empty() else 0.0
    return flags, max_flags, round(float(mean or 0.0), 2)


def build_graph(links: pl.DataFrame) -> nx.Graph:
    """Entity and attribute nodes from the link frame.

    Every row adds an edge typed by its attribute. Attribute nodes of different
    types that share a value are joined by an ``equality`` edge.
    """
    graph = nx.Graph()
    value_to_nodes: defaultdict[str, set[str]] = defaultdict(set)
    for entity, attribute, value in links.select(ENTITY, ATTRIBUTE, VALUE).rows():
        entity_node = qualify_entity(entity)
        attribute_node = qualify(attribute, value)
        graph.add_edge(*sorted((entity_node, attribute_node)), type=attribute)
        graph.add_node(entity_node, type=ENTITY_LABEL)
        graph.add_node(attribute_node, type=attribute)
        value_to_nodes[value].add(attribute_node)
    for nodes in value_to_nodes.values():
        ordered = sorted(nodes)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                graph.add_edge(first, second, type=EQUALITY_EDGE_TYPE)
    return graph


def model_summary(
    graph: nx.Graph, flags: pl.DataFrame, group_links: pl.DataFrame
) -> ModelSummary:
    entities = sum(1 for node in graph.nodes() if is_entity(node))
    flag_count = int(flags.get_column(COUNT).sum()) if not flags.is_empty() else 0
    groups = (
        group_links.select(GROUP, VALUE).unique().height
        if not group_links.is_empty()
        else 0
    )
    return ModelSummary(
        entities=entities,
        attributes=graph.number_of_nodes() - entities,
        flags=flag_count,
        groups=groups,
        links=graph.number_of_edges(),
    )
