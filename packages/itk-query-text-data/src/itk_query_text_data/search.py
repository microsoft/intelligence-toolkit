# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Embedding, query expansion and community-guided relevance search."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from itk_query_text_data import prompts
from itk_query_text_data.analysis import ThematicAnalysis, system_message
from itk_query_text_data.models import (
    ChunksTested,
    Progress,
    RelevanceTest,
    SearchConfig,
    SearchResult,
)

if TYPE_CHECKING:
    from itk_query_text_data.llm import LLM, Embedder
    from itk_query_text_data.models import ConceptGraph, Corpus, EventHandler

Embeddings = Mapping[int, Sequence[float]]

ROOT_COMMUNITY = "1"
_ROOT_LEVEL = -1
_LOGIT_BIAS = 5


async def embed_chunks(
    corpus: Corpus,
    embedder: Embedder,
    *,
    on_event: EventHandler | None = None,
) -> dict[int, list[float]]:
    """Embed every chunk's prompt text. Returns vectors keyed by chunk id."""
    if on_event is not None:
        on_event(Progress("embed", 0, len(corpus)))
    vectors = await embedder.embed([chunk.to_prompt() for chunk in corpus])
    if len(vectors) != len(corpus):
        msg = f"Embedder returned {len(vectors)} vectors for {len(corpus)} chunks"
        raise ValueError(msg)
    if on_event is not None:
        on_event(Progress("embed", len(corpus), len(corpus)))
    return {
        chunk.id: list(vector) for chunk, vector in zip(corpus, vectors, strict=True)
    }


async def rewrite_query(
    query: str,
    graph: ConceptGraph,
    llm: LLM,
    *,
    top_concepts: int = 100,
) -> str:
    """Rephrase the query using the corpus's most connected concepts, keeping its key phrases."""
    concepts = ", ".join(graph.top_concepts(top_concepts))
    return await llm.complete(
        system_message(prompts.query_anchoring_prompt, query=query, concepts=concepts)
    )


@cache
def _yes_no_logit_bias() -> dict[int, int]:
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    return {
        encoding.encode("Yes")[0]: _LOGIT_BIAS,
        encoding.encode("No")[0]: _LOGIT_BIAS,
    }


def _normalize_response(response: str) -> str:
    return "Yes" if response.strip().rstrip(".").lower() == "yes" else response


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / norm


def rank_chunks(query_vector: Sequence[float], embeddings: Embeddings) -> list[int]:
    """Chunk ids from most to least similar to the query (ties by id)."""
    target = np.asarray(query_vector, dtype=float)
    distances = {
        chunk_id: _cosine_distance(target, np.asarray(vector, dtype=float))
        for chunk_id, vector in embeddings.items()
    }
    return sorted(distances, key=lambda chunk_id: (distances[chunk_id], chunk_id))


def _community_plan(
    graph: ConceptGraph, ranking: list[int], ranking_chunks: int
) -> tuple[dict[int, list[str]], dict[int, dict[str, list[int]]]]:
    """Per level: communities in search order and the chunks assigned to each.

    Communities are ordered by the mean semantic rank of their best chunks; each
    chunk is assigned to its best-ordered community, keeping semantic order.
    """
    rank = {chunk_id: position for position, chunk_id in enumerate(ranking)}
    sequences: dict[int, list[str]] = {_ROOT_LEVEL: [ROOT_COMMUNITY]}
    assignments: dict[int, dict[str, list[int]]] = {
        _ROOT_LEVEL: {ROOT_COMMUNITY: list(ranking)}
    }
    for level in range(graph.max_level + 1):
        chunk_communities: dict[int, set[str]] = defaultdict(set)
        community_chunks: dict[str, list[int]] = defaultdict(list)
        for chunk_id in ranking:
            for concept in graph.chunk_concepts.get(chunk_id, ()):
                community = graph.community_of(concept, level)
                if community is not None:
                    chunk_communities[chunk_id].add(community)
            for community in sorted(chunk_communities[chunk_id]):
                community_chunks[community].append(chunk_id)
        mean_rank = {
            community: float(np.mean([rank[c] for c in chunk_ids[:ranking_chunks]]))
            for community, chunk_ids in community_chunks.items()
        }
        sequence = sorted(mean_rank, key=lambda c: (mean_rank[c], c))
        order = {community: position for position, community in enumerate(sequence)}
        assigned: dict[str, list[int]] = defaultdict(list)
        for chunk_id in ranking:
            communities = chunk_communities.get(chunk_id)
            if communities:
                assigned[min(communities, key=order.__getitem__)].append(chunk_id)
        sequences[level] = sequence
        assignments[level] = dict(assigned)
    return sequences, assignments


class _Search:
    def __init__(
        self,
        corpus: Corpus,
        llm: LLM,
        query: str,
        config: SearchConfig,
        analysis: ThematicAnalysis,
        on_event: EventHandler | None,
    ) -> None:
        self.corpus = corpus
        self.llm = llm
        self.query = query
        self.config = config
        self.analysis = analysis
        self.on_event = on_event
        self.tests: list[RelevanceTest] = []
        self.seen: set[int] = set()

    def adjacent(self) -> list[int]:
        targets: set[int] = set()
        for test in self.tests:
            if test.relevant:
                targets |= self.corpus.neighbours(
                    test.chunk_id, self.config.adjacent_test_steps
                )
        return sorted(targets - self.seen)

    def budget_used(self) -> int:
        return len(self.tests) + len(self.adjacent())

    async def _judge(self, chunk_id: int) -> str:
        return await self.llm.complete(
            system_message(
                prompts.chunk_relevance_prompt,
                chunk=self.corpus[chunk_id].to_prompt(),
                query=self.query,
            ),
            max_tokens=1,
            logit_bias=_yes_no_logit_bias(),
        )

    async def assess(self, label: str, chunk_ids: list[int], num_adjacent: int) -> bool:
        """Test chunks in batches, stopping after a batch with no relevant chunk."""
        size = self.config.relevance_test_batch_size
        budget = self.config.relevance_test_budget
        found = False
        for start in range(0, len(chunk_ids), size):
            batch = chunk_ids[start : start + size]
            if len(self.tests) + len(batch) + num_adjacent > budget:
                batch = batch[: max(budget - len(self.tests), 0)]
            responses = await asyncio.gather(*(self._judge(c) for c in batch))
            newly_relevant = []
            for chunk_id, response in zip(batch, responses, strict=True):
                if chunk_id in self.seen:
                    continue
                test = RelevanceTest(label, chunk_id, _normalize_response(response))
                self.tests.append(test)
                self.seen.add(chunk_id)
                if test.relevant:
                    newly_relevant.append(chunk_id)
            if self.on_event is not None:
                self.on_event(ChunksTested(tuple(self.tests)))
            if newly_relevant:
                await self.analysis.add(newly_relevant)
            found = bool(newly_relevant)
            if not found:
                break
        return found

    async def run(
        self,
        sequences: dict[int, list[str]],
        assignments: dict[int, dict[str, list[int]]],
        parents: dict[str, str | None],
    ) -> None:
        config = self.config
        if self.on_event is not None:
            self.on_event(ChunksTested(()))
        level = _ROOT_LEVEL
        eliminated: set[str] = set()
        successive_irrelevant = 0
        while self.budget_used() < config.relevance_test_budget:
            relevant_this_pass = False
            tested_this_pass = False
            sequence = []
            for community in sequences[level]:
                if parents.get(community) in eliminated:
                    eliminated.add(community)
                else:
                    sequence.append(community)
            for community in sequence:
                unseen = [
                    c
                    for c in assignments[level].get(community, [])
                    if c not in self.seen
                ][: config.community_relevance_tests]
                if not unseen:
                    continue
                tested_this_pass = True
                is_relevant = await self.assess(
                    f"topic {community}", unseen, len(self.adjacent())
                )
                if self.budget_used() >= config.relevance_test_budget:
                    break
                relevant_this_pass |= is_relevant
                if level > _ROOT_LEVEL and not is_relevant:
                    eliminated.add(community)
                    successive_irrelevant += 1
                    if successive_irrelevant == config.irrelevant_community_restart:
                        successive_irrelevant = 0
                        break
                else:
                    successive_irrelevant = 0
            if not tested_this_pass:
                break
            if level > _ROOT_LEVEL and not relevant_this_pass:
                break
            if level + 1 in sequences:
                level += 1
        adjacent = self.adjacent()
        await self.assess("neighbours", adjacent, len(adjacent))
        await self.analysis.complete()


async def search(
    query: str,
    corpus: Corpus,
    graph: ConceptGraph,
    embeddings: Embeddings,
    *,
    llm: LLM,
    embedder: Embedder,
    expanded_query: str | None = None,
    config: SearchConfig | None = None,
    on_event: EventHandler | None = None,
) -> SearchResult:
    """Find chunks relevant to a query within an LLM relevance-test budget.

    Chunks are ranked by embedding similarity to ``expanded_query`` (default:
    ``query``). The search tests the best matches overall, then descends the
    concept-community hierarchy, testing communities in order of semantic rank
    and skipping the children of communities found irrelevant. Finally it tests
    neighbours of relevant chunks. ``query`` itself is used for the live
    thematic analysis.
    """
    config = config or SearchConfig()
    expanded_query = expanded_query or query
    query_vector = (await embedder.embed([expanded_query]))[0]
    ranking = rank_chunks(
        query_vector, {c: v for c, v in embeddings.items() if c in corpus}
    )
    sequences, assignments = _community_plan(
        graph, ranking, config.community_ranking_chunks
    )
    parents = {community.id: community.parent for community in graph.communities}

    analysis = ThematicAnalysis(
        query, corpus, llm, config.analysis_update_interval, on_event
    )
    runner = _Search(corpus, llm, expanded_query, config, analysis, on_event)
    await runner.run(sequences, assignments, parents)

    relevant = sorted(test.chunk_id for test in runner.tests if test.relevant)
    return SearchResult(
        query=query,
        expanded_query=expanded_query,
        tests=tuple(runner.tests),
        relevant_chunk_ids=tuple(relevant),
        themes=analysis.clustered_ids(),
        analysis=analysis.structure if analysis.enabled else None,
        analysis_markdown=analysis.markdown() if analysis.enabled else "",
    )


def format_search_summary(tests: Sequence[RelevanceTest]) -> str:
    """Markdown summary of relevant/tested counts per search step; empty steps in red."""
    rounds: list[str] = []
    label: str | None = None
    relevant = tested = 0

    def close_round() -> None:
        text = f"{label}: {relevant}/{tested}"
        rounds.append(text if relevant else f"<span style='color: red'>{text}</span>")

    for test in tests:
        if test.label != label:
            if label is not None:
                close_round()
            label, relevant, tested = test.label, 0, 0
        tested += 1
        relevant += test.relevant
    if label is not None:
        close_round()
    total_relevant = sum(test.relevant for test in tests)
    summary = f"**Relevant chunks / tested chunks: {total_relevant}/{len(tests)}**"
    if rounds:
        summary += " (" + "; ".join(rounds) + ")"
    return summary
