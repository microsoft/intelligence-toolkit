# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


import io
from typing import ClassVar

import numpy as np
import polars as pl

import intelligence_toolkit.AI.utils as utils
from intelligence_toolkit.AI.classes import LLMCallback
from intelligence_toolkit.AI.client import OpenAIClient
from intelligence_toolkit.helpers.classes import IntelligenceWorkflow
from intelligence_toolkit.match_entity_records import prompts
from intelligence_toolkit.match_entity_records.classes import (
    AttributeToMatch,
    RecordsModel,
)
from intelligence_toolkit.match_entity_records.detect import (
    build_attributes_dataframe,
    build_matches,
    build_matches_dataset,
    build_near_map,
    build_nearest_neighbors,
    build_sentence_pair_scores,
    convert_to_sentences,
)
from intelligence_toolkit.match_entity_records.prepare_model import (
    build_attribute_options,
    build_attributes_list,
    format_model_df,
)


class MatchEntityRecords(IntelligenceWorkflow):
    model_dfs: ClassVar[dict] = {}
    max_rows_to_process = 0
    evaluations_df = pl.DataFrame()
    matches_df = pl.DataFrame()

    @property
    def total_records(self) -> int:
        return sum(df.shape[0] for df in self.model_dfs.values())

    @property
    def attribute_options(self) -> str:
        return build_attribute_options(self.model_dfs)

    @property
    def integrated_results(self) -> pl.DataFrame:
        value = self.evaluations_df.drop_nulls()
        return self.matches_df.join(value, on="Group ID", how="inner")

    def add_df_to_model(self, model: RecordsModel) -> pl.DataFrame:
        if not model.dataframe_name:
            model.dataframe_name = "dataset_" + len(self.model_dfs) + 1

        self.model_dfs[model.dataframe_name] = format_model_df(
            model,
            self.max_rows_to_process,
        )
        return self.model_dfs[model.dataframe_name]

    def build_model_df(self, attributes_list: list[AttributeToMatch]) -> pl.DataFrame:
        attributes = build_attributes_list(attributes_list)
        self.model_df = build_attributes_dataframe(self.model_dfs, attributes)
        self.model_df = self.model_df.with_columns(
            (pl.col("Entity ID").cast(pl.Utf8))
            + "::"
            + pl.col("Dataset").alias("Unique ID")
        )

        self.sentences_vector_data = convert_to_sentences(self.model_df)
        return self.model_df

    async def embed_sentences(self) -> None:
        sentences_data = await self.embedder.embed_store_many(
            self.sentences_vector_data, cache_data=self.cache_embeddings
        )
        self.all_sentences = [x["text"] for x in self.sentences_vector_data]
        self.embeddings = [
            np.array(next(d["vector"] for d in sentences_data if d["text"] == f))
            for f in self.all_sentences
        ]

    def detect_record_groups(
        self, pair_embedding_threshold: int, pair_jaccard_threshold: int
    ) -> pl.DataFrame:
        distances, indices = build_nearest_neighbors(self.embeddings)
        near_map = build_near_map(
            distances,
            indices,
            self.all_sentences,
            pair_embedding_threshold,
        )

        pair_scores = build_sentence_pair_scores(near_map, self.model_df)

        entity_to_group, matches, pair_to_match = build_matches(
            pair_scores,
            self.model_df,
            pair_jaccard_threshold,
        )

        matches_df = pl.DataFrame(
            list(matches),
            schema=["Group ID", *self.model_df.columns],
        ).sort(by=["Group ID", "Entity name", "Dataset"], descending=False)

        self.matches_df = build_matches_dataset(
            matches_df, pair_to_match, entity_to_group
        )
        return self.matches_df

    async def evaluate_groups(
        self,
        ai_instructions=prompts.list_prompts,
        callbacks: list[LLMCallback] | None = None,
    ) -> None:
        data = self.model_df.drop(
            [
                "Entity ID",
                "Dataset",
                "Name similarity",
            ]
        ).to_pandas()
        data = data.head(500)

        batch_messages = utils.generate_batch_messages(
            ai_instructions, batch_name="data", batch_value=data
        )
        prefix = "```\nGroup ID,Relatedness,Explanation\n"

        for messages in batch_messages:
            response = await OpenAIClient(self.ai_configuration).generate_chat_async(
                messages, stream=True
            )
            prefix = prefix + response + "\n"
        result = prefix.replace("```\n", "").strip()
        self.evaluations_df = pl.read_csv(io.StringIO(result))
        return result

    def clear_model_dfs(self) -> None:
        self.model_dfs = {}
