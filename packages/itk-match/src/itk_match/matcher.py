# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Facade that runs the pipeline end to end."""

from __future__ import annotations

from typing import TYPE_CHECKING

from itk_core.client import OpenAIClient
from itk_core.local_embedder import LocalEmbedder
from itk_core.openai_embedder import OpenAIEmbedder

from itk_match.blocking import candidate_pairs
from itk_match.cluster import cluster
from itk_match.embed import embed_sentences
from itk_match.evaluate import evaluate_groups
from itk_match.models import GROUP_ID, NAME, Dataset, MatchConfig, MatchResult
from itk_match.names import get_scorer, score_pairs
from itk_match.sentences import to_sentences
from itk_match.unify import unify

if TYPE_CHECKING:
    import polars as pl
    from itk_core.classes import LLMCallback
    from itk_core.openai_configuration import OpenAIConfiguration
    from itk_core.progress import ProgressBatchCallback

    from itk_match.embed import Embedder
    from itk_match.evaluate import ChatFn


class Matcher:
    """Holds the embedder and chat function; everything else is passed in and returned."""

    def __init__(self, embedder: Embedder, chat: ChatFn | None = None) -> None:
        self.embedder = embedder
        self.chat = chat

    @classmethod
    def from_openai(
        cls, configuration: OpenAIConfiguration, local_embeddings: bool = False
    ) -> Matcher:
        embedder = (
            LocalEmbedder() if local_embeddings else OpenAIEmbedder(configuration)
        )
        return cls(embedder, OpenAIClient(configuration).generate_chat_async)

    async def match(
        self,
        datasets: list[Dataset],
        config: MatchConfig | None = None,
        callbacks: list[ProgressBatchCallback] | None = None,
    ) -> MatchResult:
        """Unify, embed, block, score and cluster the given datasets."""
        config = config or MatchConfig()
        records = unify(datasets, config.max_rows_per_dataset)
        embeddings = await embed_sentences(
            to_sentences(records), self.embedder, callbacks
        )
        return self.regroup(MatchResult(records, [], [], config, embeddings), config)

    @staticmethod
    def regroup(result: MatchResult, config: MatchConfig) -> MatchResult:
        """Re-run blocking, scoring and clustering with new thresholds, no re-embedding."""
        if result.embeddings is None:
            msg = "MatchResult has no embeddings to regroup from"
            raise ValueError(msg)
        candidates = candidate_pairs(
            result.embeddings, config.max_distance, config.neighbors
        )
        names = result.records.get_column(NAME).to_list()
        pairs = score_pairs(candidates, names, get_scorer(config.name_scorer))
        groups = cluster(pairs, config.min_name_similarity)
        return MatchResult(result.records, pairs, groups, config, result.embeddings)

    async def evaluate(
        self,
        result: MatchResult,
        groups_per_batch: int = 20,
        callbacks: list[LLMCallback] | None = None,
    ) -> pl.DataFrame:
        """Ask the LLM to rate each group; requires a chat function."""
        if self.chat is None:
            msg = "Matcher has no chat function; use Matcher.from_openai"
            raise ValueError(msg)
        return await evaluate_groups(
            result.groups_frame, self.chat, groups_per_batch, callbacks
        )

    @staticmethod
    def integrate(result: MatchResult, evaluations: pl.DataFrame) -> pl.DataFrame:
        """Matched records joined with their group's evaluation."""
        return result.groups_frame.join(evaluations, on=GROUP_ID, how="left")
