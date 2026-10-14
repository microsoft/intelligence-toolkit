# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Typed records passed between the workflow steps."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class Document:
    """A named text before chunking. ``period`` groups documents by time (e.g. ``"2024-Q1"``)."""

    source: str
    text: str
    period: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A piece of a document.

    ``id`` is unique across the corpus (1-based, in corpus order); ``index`` is the
    1-based position of the chunk within its source.
    """

    id: int
    source: str
    index: int
    text: str
    period: str | None = None

    @property
    def reference(self) -> str:
        return f"{self.source} ({self.index})"

    def to_prompt(self) -> str:
        """Render the chunk as the JSON record shown to the LLM and embedded."""
        record: dict[str, Any] = {
            "title": self.source,
            "text_chunk": self.text,
            "chunk_id": self.index,
        }
        if self.period is not None:
            record["period"] = self.period
        return json.dumps(record, indent=2, ensure_ascii=False)


@dataclass(frozen=True, eq=False)
class Corpus:
    """An ordered, immutable collection of chunks with neighbour lookups."""

    chunks: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        ids = [chunk.id for chunk in self.chunks]
        if ids != list(range(1, len(ids) + 1)):
            msg = "Chunk ids must be 1..n in corpus order"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    def __getitem__(self, chunk_id: int) -> Chunk:
        if not 1 <= chunk_id <= len(self.chunks):
            raise KeyError(chunk_id)
        return self.chunks[chunk_id - 1]

    def __contains__(self, chunk_id: object) -> bool:
        return isinstance(chunk_id, int) and 1 <= chunk_id <= len(self.chunks)

    @cached_property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(chunk.source for chunk in self.chunks))

    @cached_property
    def periods(self) -> dict[str, tuple[int, ...]]:
        """Chunk ids per period, for chunks that have one."""
        result: dict[str, list[int]] = {}
        for chunk in self.chunks:
            if chunk.period is not None:
                result.setdefault(chunk.period, []).append(chunk.id)
        return {period: tuple(ids) for period, ids in result.items()}

    def previous_id(self, chunk_id: int) -> int | None:
        if chunk_id - 1 in self and self[chunk_id - 1].source == self[chunk_id].source:
            return chunk_id - 1
        return None

    def next_id(self, chunk_id: int) -> int | None:
        if chunk_id + 1 in self and self[chunk_id + 1].source == self[chunk_id].source:
            return chunk_id + 1
        return None

    def neighbours(self, chunk_id: int, steps: int) -> set[int]:
        """Chunks up to ``steps`` positions before or after ``chunk_id`` in the same source."""
        result: set[int] = set()
        for step in (self.previous_id, self.next_id):
            current: int | None = chunk_id
            for _ in range(steps):
                current = step(current)
                if current is None:
                    break
                result.add(current)
        return result

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "source": chunk.source,
                "index": chunk.index,
                "text": chunk.text,
                "period": chunk.period,
            }
            for chunk in self.chunks
        ]

    @classmethod
    def from_records(cls, records: list[Mapping[str, Any]]) -> Corpus:
        chunks = []
        for position, record in enumerate(records, start=1):
            period = record.get("period")
            chunks.append(
                Chunk(
                    id=position,
                    source=str(record["source"]),
                    index=int(record["index"]),
                    text=str(record["text"]),
                    period=None if period is None or period != period else str(period),
                )
            )
        return cls(tuple(chunks))


HUB_NODE = "dummynode"
"""Synthetic node that links graph components before community detection."""


@dataclass(frozen=True)
class Community:
    """A concept community. ``id`` encodes the hierarchy, e.g. ``"1.2.1"`` is a child of ``"1.2"``."""

    id: str
    level: int
    parent: str | None
    concepts: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class ConceptGraph:
    """Noun-phrase co-occurrence graph and its hierarchical communities.

    ``graph`` is the pruned graph over all chunks: nodes carry ``count`` and their
    final ``community`` id; edges carry ``weight``. Treat it as read-only.
    """

    graph: nx.Graph
    communities: tuple[Community, ...]
    chunk_concepts: Mapping[int, tuple[str, ...]]
    concept_chunks: Mapping[str, tuple[int, ...]]
    period_graphs: Mapping[str, nx.Graph] = field(default_factory=dict)

    @cached_property
    def max_level(self) -> int:
        """Deepest community level, or -1 when there are no communities."""
        return max((c.level for c in self.communities), default=-1)

    @cached_property
    def _concept_levels(self) -> dict[str, dict[int, str]]:
        result: dict[str, dict[int, str]] = {}
        for community in self.communities:
            for concept in community.concepts:
                result.setdefault(concept, {})[community.level] = community.id
        return result

    def community_of(self, concept: str, level: int) -> str | None:
        """Community of ``concept`` at ``level``, or at its deepest level above it.

        A concept stops appearing at deeper levels once its community is small
        enough not to be split, so it stays in that community from then on.
        """
        levels = self._concept_levels.get(concept)
        if not levels:
            return None
        available = [lvl for lvl in levels if lvl <= level]
        return levels[max(available)] if available else None

    def final_community(self, concept: str) -> str | None:
        return self.community_of(concept, self.max_level)

    def top_concepts(self, limit: int) -> list[str]:
        """Concepts ordered by degree (ties alphabetical)."""
        ranked = sorted(self.graph.degree(), key=lambda nd: (-nd[1], nd[0]))
        return [node for node, _ in ranked[:limit]]

    def community_networks(self) -> dict[int, dict[str, nx.Graph]]:
        """Subgraph per community, keyed by level then by a label of its top concepts.

        Communities are ordered by size (largest first); labels join the seven
        highest-degree concepts.
        """
        result: dict[int, dict[str, nx.Graph]] = {}
        ordered = sorted(
            self.communities, key=lambda c: (c.level, -len(c.concepts), c.id)
        )
        for community in ordered:
            top = sorted(community.concepts, key=lambda n: (-self.graph.degree(n), n))[
                :7
            ]
            result.setdefault(community.level, {})["; ".join(top)] = (
                self.graph.subgraph(community.concepts)
            )
        return result


@dataclass(frozen=True)
class SearchConfig:
    """Settings for :func:`itk_query_text_data.search`.

    Attributes
    ----------
    relevance_test_budget:
        Maximum number of LLM relevance tests per query.
    relevance_test_batch_size:
        Tests run concurrently per batch.
    community_relevance_tests:
        Tests per community visit.
    community_ranking_chunks:
        Top semantic matches used to rank each community.
    irrelevant_community_restart:
        Successive irrelevant communities before restarting the level in relevance order.
    adjacent_test_steps:
        Neighbouring chunks (before and after) of each relevant chunk to test at the end.
    analysis_update_interval:
        Relevant chunks between live thematic analysis updates; 0 disables them.
    """

    relevance_test_budget: int = 20
    relevance_test_batch_size: int = 5
    community_relevance_tests: int = 5
    community_ranking_chunks: int = 5
    irrelevant_community_restart: int = 5
    adjacent_test_steps: int = 1
    analysis_update_interval: int = 10

    def __post_init__(self) -> None:
        for name in (
            "relevance_test_budget",
            "relevance_test_batch_size",
            "community_relevance_tests",
            "community_ranking_chunks",
            "irrelevant_community_restart",
        ):
            if getattr(self, name) < 1:
                msg = f"{name} must be at least 1"
                raise ValueError(msg)
        if self.adjacent_test_steps < 0 or self.analysis_update_interval < 0:
            msg = "adjacent_test_steps and analysis_update_interval must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class RelevanceTest:
    """One LLM relevance judgement. ``label`` names the search step, e.g. ``"topic 1.2"``."""

    label: str
    chunk_id: int
    response: str

    @property
    def relevant(self) -> bool:
        return self.response == "Yes"


@dataclass(frozen=True, eq=False)
class ThemeStructure:
    """Live thematic analysis: points with source chunk ids, grouped into themes."""

    points: Mapping[Any, str]
    point_sources: Mapping[Any, tuple[int, ...]]
    themes: Mapping[str, tuple[Any, ...]]


@dataclass(frozen=True, eq=False)
class SearchResult:
    query: str
    expanded_query: str
    tests: tuple[RelevanceTest, ...]
    relevant_chunk_ids: tuple[int, ...]
    themes: Mapping[str, tuple[int, ...]]
    """Relevant chunk ids grouped by theme; one ``"All relevant chunks"`` group without live analysis."""
    analysis: ThemeStructure | None = None
    analysis_markdown: str = ""
    """Rendered live analysis (empty when it was disabled)."""

    @property
    def summary(self) -> str:
        from itk_query_text_data.search import format_search_summary

        return format_search_summary(self.tests)


@dataclass(frozen=True, eq=False)
class Answer:
    markdown: str
    references: tuple[int, ...]
    """Cited chunk ids that exist in the corpus, ascending."""
    report: Mapping[str, str]
    """The integrated report fields: answer, report_title, report_overview, report_implications."""
    themes: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Progress:
    step: str
    current: int
    total: int


@dataclass(frozen=True)
class ChunksTested:
    tests: tuple[RelevanceTest, ...]
    """Every test so far, in order."""

    @property
    def relevant_chunk_ids(self) -> list[int]:
        return [test.chunk_id for test in self.tests if test.relevant]


@dataclass(frozen=True)
class AnalysisUpdated:
    structure: ThemeStructure
    markdown: str


Event = Progress | ChunksTested | AnalysisUpdated
EventHandler = Callable[[Event], None]
TokenHandler = Callable[[str], None]
