# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Per-entity records with the flag statistics of their network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from itk_networks.models import (
    COUNT,
    ENTITY,
    ENTITY_RECORD_COLUMNS,
    ENTITY_RECORD_SCHEMA,
    GROUP,
    QUALIFIED_ENTITY,
    VALUE,
    EntityRecord,
)
from itk_networks.text import node_value

if TYPE_CHECKING:
    from itk_networks.models import Network

Links = dict[str, frozenset[str]]


def flag_totals(flags: pl.DataFrame) -> dict[str, int]:
    """Total flag count per qualified entity."""
    if flags.is_empty():
        return {}
    totals = flags.group_by(QUALIFIED_ENTITY).agg(pl.col(COUNT).sum())
    return dict(
        zip(
            totals.get_column(QUALIFIED_ENTITY).to_list(),
            (int(v) for v in totals.get_column(COUNT).to_list()),
            strict=True,
        )
    )


def network_flags(
    entities: tuple[str, ...], totals: dict[str, int], links: Links
) -> tuple[int, int, float, float, int]:
    """``(network_flags, flagged, flagged_per_unflagged, flags_per_entity, size)``.

    Entities reached through inferred links are counted into the network's
    totals once each, exactly as the legacy ``get_integrated_flags`` did.
    """
    size = len(entities)
    if not totals:
        return 0, 0, 0.0, 0.0, size
    seen = set(entities)
    total = sum(totals.get(e, 0) for e in entities)
    flagged = sum(1 for e in entities if totals.get(e, 0) > 0)
    for entity in entities:
        for other in sorted(links.get(entity, frozenset())):
            if other in seen:
                continue
            seen.add(other)
            count = totals.get(other, 0)
            total += count
            size += 1
            if count > 0:
                flagged += 1
    unflagged = size - flagged
    per_unflagged = round(flagged / unflagged, 2) if unflagged > 0 else 0.0
    per_entity = round(total / size, 2) if size > 0 else 0.0
    return total, flagged, per_unflagged, per_entity, size


def entity_records(
    networks: list[Network], flags: pl.DataFrame, links: Links | None = None
) -> list[EntityRecord]:
    totals = flag_totals(flags)
    links = links or {}
    records: list[EntityRecord] = []
    for network in networks:
        total, flagged, per_unflagged, per_entity, size = network_flags(
            network.entities, totals, links
        )
        records.extend(
            EntityRecord(
                entity_id=node_value(entity),
                entity_flags=totals.get(entity, 0),
                network_id=network.id,
                network_entities=size,
                network_flags=total,
                flagged=flagged,
                flags_per_entity=per_entity,
                flagged_per_unflagged=per_unflagged,
            )
            for entity in network.entities
        )
    return records


def entities_frame(records: list[EntityRecord]) -> pl.DataFrame:
    """One row per entity, most flagged-per-unflagged networks first."""
    rows = [r.as_row() for r in records]
    frame = pl.DataFrame(
        {col: [row[i] for row in rows] for i, col in enumerate(ENTITY_RECORD_COLUMNS)},
        schema=ENTITY_RECORD_SCHEMA,
    )
    # Stable on the locked polars (0.20.10 has no maintain_order on sort).
    return (
        frame.with_row_index("_row")
        .sort(["flagged/unflagged", "_row"], descending=[True, False])
        .drop("_row")
    )


def grouped_frame(entities: pl.DataFrame, group_links: pl.DataFrame) -> pl.DataFrame:
    """Entity rows joined with one column per group dimension (first value wins)."""
    out = entities
    if group_links.is_empty():
        return out
    for group in group_links.get_column(GROUP).unique(maintain_order=True).to_list():
        pivot = (
            group_links.filter(pl.col(GROUP) == group)
            .pivot(
                values=VALUE, index=ENTITY, columns=GROUP, aggregate_function="first"
            )
            .rename({ENTITY: "entity_id"})
        )
        out = out.join(pivot, on="entity_id", how="left")
    return out
