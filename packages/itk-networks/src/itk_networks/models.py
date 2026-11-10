# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Typed inputs and outputs of the network detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import networkx as nx
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from itk_networks.text import ENTITY_LABEL, is_entity

if TYPE_CHECKING:
    import numpy as np

# Column names of the link frames produced by itk_networks.prepare.
ENTITY = "entity"
ATTRIBUTE = "attribute"
VALUE = "value"
TYPE = "type"
FLAG = "flag"
COUNT = "count"
GROUP = "group"
QUALIFIED_ENTITY = "qualified_entity"

ATTRIBUTE_LINK_SCHEMA = {ENTITY: pl.Utf8, ATTRIBUTE: pl.Utf8, VALUE: pl.Utf8}
FLAG_LINK_SCHEMA = {ENTITY: pl.Utf8, TYPE: pl.Utf8, FLAG: pl.Utf8, COUNT: pl.Int64}
GROUP_LINK_SCHEMA = {ENTITY: pl.Utf8, GROUP: pl.Utf8, VALUE: pl.Utf8}
FLAGS_SCHEMA = {**FLAG_LINK_SCHEMA, QUALIFIED_ENTITY: pl.Utf8}
TRIMMED_SCHEMA = {"Attribute": pl.Utf8, "Linked Entities": pl.Int64}

# Column names of the entity records frame, kept from the legacy workflow.
ENTITY_RECORD_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "entity_flags",
    "network_id",
    "network_entities",
    "network_flags",
    "flagged",
    "flags/entity",
    "flagged/unflagged",
)
ENTITY_RECORD_SCHEMA = {
    "entity_id": pl.Utf8,
    "entity_flags": pl.Int64,
    "network_id": pl.Int64,
    "network_entities": pl.Int64,
    "network_flags": pl.Int64,
    "flagged": pl.Int64,
    "flags/entity": pl.Float64,
    "flagged/unflagged": pl.Float64,
}

DEFAULT_MAX_ATTRIBUTE_DEGREE = 10
DEFAULT_MAX_NETWORK_ENTITIES = 20
DEFAULT_NEIGHBORS = 20
DEFAULT_LEIDEN_SEED = 42
SIMILARITY_THRESHOLD_MIN = 0.001
SIMILARITY_THRESHOLD_MAX = 1.0


class FlagFormat(str, Enum):
    """How flag columns are read.

    ``Count``: each column is a flag type and its value a count per row.
    ``Instance``: each row is one occurrence; the summed value names the flag.
    Both follow the legacy ``FlagAggregatorType`` semantics exactly.
    """

    Instance = "Instance"
    Count = "Count"


class _LinkedTable(BaseModel):
    """A frame, the column identifying the entity, and the columns to link."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frame: pl.DataFrame
    entity_column: str
    columns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_columns(self) -> _LinkedTable:
        wanted = {self.entity_column, *self.columns}
        missing = sorted(wanted - set(self.frame.columns))
        if missing:
            msg = f"{type(self).__name__} has no columns {missing}"
            raise ValueError(msg)
        if self.entity_column in self.columns:
            msg = f"Entity column {self.entity_column!r} cannot also be linked"
            raise ValueError(msg)
        if ENTITY_LABEL in self.columns:
            msg = f"{ENTITY_LABEL!r} is reserved for entity nodes"
            raise ValueError(msg)
        return self


class AttributeTable(_LinkedTable):
    """Entity rows whose ``columns`` become attribute nodes linked to the entity."""


class FlagTable(_LinkedTable):
    """Entity rows whose ``columns`` carry flags, read according to ``format``."""

    format: FlagFormat = FlagFormat.Count


class GroupTable(_LinkedTable):
    """Entity rows whose ``columns`` are grouping dimensions for the output frame."""


class NetworkConfig(BaseModel):
    """Knobs for identifying networks from the prepared graph."""

    model_config = ConfigDict(frozen=True)

    max_attribute_degree: int = Field(
        DEFAULT_MAX_ATTRIBUTE_DEGREE,
        ge=0,
        description="Attributes shared by more entities than this are ignored",
    )
    max_network_entities: int = Field(
        DEFAULT_MAX_NETWORK_ENTITIES,
        ge=1,
        description="Components larger than this are split by the community detector",
    )
    supporting_attribute_types: tuple[str, ...] = Field(
        (),
        description="Attribute types that may not, on their own, connect entities",
    )
    removed_attributes: tuple[str, ...] = Field(
        (), description="Attribute nodes the user chose to ignore"
    )
    leiden_seed: int = Field(DEFAULT_LEIDEN_SEED, description="Seed for the detector")


@dataclass(frozen=True, slots=True)
class ModelSummary:
    entities: int
    attributes: int
    flags: int
    groups: int
    links: int

    def __str__(self) -> str:
        return (
            f"Number of entities: {self.entities}, "
            f"Number of attributes: {self.attributes}, "
            f"Number of flags: {self.flags}, "
            f"Number of groups: {self.groups}, "
            f"Number of links: {self.links}"
        )


@dataclass(frozen=True, slots=True)
class Network:
    """Entities (qualified node names) believed to form one network."""

    id: int
    entities: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.entities)


@dataclass(frozen=True, slots=True)
class EntityRecord:
    """One entity with the flag statistics of its network."""

    entity_id: str
    entity_flags: int
    network_id: int
    network_entities: int
    network_flags: int
    flagged: int
    flags_per_entity: float
    flagged_per_unflagged: float

    def as_row(self) -> tuple[str, int, int, int, int, int, float, float]:
        return (
            self.entity_id,
            self.entity_flags,
            self.network_id,
            self.network_entities,
            self.network_flags,
            self.flagged,
            self.flags_per_entity,
            self.flagged_per_unflagged,
        )


@dataclass(frozen=True, slots=True)
class ExposureSummary:
    direct: int
    indirect: int
    paths: int
    entities: int


@dataclass(frozen=True, slots=True)
class ExposureNode:
    """A node on an exposure path: entities carry flags, attributes a degree."""

    node: str
    flags: int | None = None
    entities: int | None = None


@dataclass(frozen=True, slots=True)
class ExposurePath:
    """Steps from one or more flagged sources to the selected entity.

    ``steps[0]`` holds the sources that share the remaining route; every later
    step holds exactly one node.
    """

    steps: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class Exposure:
    entity: str
    summary: ExposureSummary
    paths: tuple[ExposurePath, ...]
    nodes: tuple[ExposureNode, ...]

    @property
    def markdown(self) -> str:
        from itk_networks.exposure import render_exposure

        return render_exposure(self)


@dataclass(slots=True)
class NetworkResult:
    """Everything produced so far by one detection run.

    Stages append to this; expensive state (the graph, embeddings and neighbour
    matrices) is kept so later stages can be re-run without redoing them.
    """

    graph: nx.Graph
    attribute_links: pl.DataFrame
    flag_links: pl.DataFrame
    group_links: pl.DataFrame
    flags: pl.DataFrame
    max_entity_flags: int
    mean_flagged_flags: float
    # index
    indexed_node_types: tuple[str, ...] = ()
    indexed_texts: tuple[str, ...] = ()
    embeddings: np.ndarray | None = field(default=None, repr=False)
    neighbor_distances: np.ndarray | None = field(default=None, repr=False)
    neighbor_indices: np.ndarray | None = field(default=None, repr=False)
    # infer
    similarity_threshold: float | None = None
    inferred_links: dict[str, frozenset[str]] = field(default_factory=dict)
    # identify
    config: NetworkConfig | None = None
    trimmed_attributes: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=TRIMMED_SCHEMA)
    )
    trimmed_nodes: frozenset[str] = frozenset()
    networks: list[Network] = field(default_factory=list)
    membership: dict[str, int] = field(default_factory=dict)
    entity_records: list[EntityRecord] = field(default_factory=list)

    @property
    def node_types(self) -> tuple[str, ...]:
        """Attribute types present in the graph, sorted."""
        return tuple(
            sorted(
                {data[TYPE] for _, data in self.graph.nodes(data=True)} - {ENTITY_LABEL}
            )
        )

    @property
    def entity_types(self) -> list[str]:
        """``ENTITY`` plus every attribute type, sorted; the indexable types."""
        return sorted([ENTITY_LABEL, *self.node_types])

    @property
    def attributes(self) -> list[str]:
        return [node for node in self.graph.nodes() if not is_entity(node)]

    @property
    def summary(self) -> ModelSummary:
        from itk_networks.prepare import model_summary

        return model_summary(self.graph, self.flags, self.group_links)

    @property
    def inferred_links_frame(self) -> pl.DataFrame:
        from itk_networks.neighbors import links_frame

        return links_frame(self.inferred_links)

    @property
    def entities_frame(self) -> pl.DataFrame:
        from itk_networks.records import entities_frame

        return entities_frame(self.entity_records)

    @property
    def grouped_frame(self) -> pl.DataFrame:
        from itk_networks.records import grouped_frame

        return grouped_frame(self.entities_frame, self.group_links)

    @property
    def network_sizes(self) -> list[int]:
        return [n.size for n in self.networks if n.size > 1]

    @property
    def records_summary(self) -> str:
        if not self.networks:
            return ""
        sizes = self.network_sizes
        largest = max(sizes) if sizes else 0
        return (
            f"Networks identified: {len(self.networks)} "
            f"({len(sizes)} with multiple entities, maximum {largest})"
        )

    def network(self, network_id: int) -> Network:
        try:
            return self.networks[network_id]
        except IndexError:
            msg = f"No network {network_id}; {len(self.networks)} identified"
            raise ValueError(msg) from None


def cast_frame(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Select the schema's columns, in order, cast to its dtypes."""
    return frame.select([pl.col(c).cast(t) for c, t in schema.items()])


def empty_result() -> NetworkResult:
    return NetworkResult(
        graph=nx.Graph(),
        attribute_links=pl.DataFrame(schema=ATTRIBUTE_LINK_SCHEMA),
        flag_links=pl.DataFrame(schema=FLAG_LINK_SCHEMA),
        group_links=pl.DataFrame(schema=GROUP_LINK_SCHEMA),
        flags=pl.DataFrame(schema=FLAGS_SCHEMA),
        max_entity_flags=0,
        mean_flagged_flags=0.0,
    )


__all__ = [
    "AttributeTable",
    "EntityRecord",
    "Exposure",
    "ExposureNode",
    "ExposurePath",
    "ExposureSummary",
    "FlagFormat",
    "FlagTable",
    "GroupTable",
    "ModelSummary",
    "Network",
    "NetworkConfig",
    "NetworkResult",
    "cast_frame",
    "empty_result",
]
