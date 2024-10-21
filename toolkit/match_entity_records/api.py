# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


from typing import ClassVar

import numpy as np
import polars as pl

from toolkit.AI import OpenAIEmbedder, utils
from toolkit.AI.client import OpenAIClient
from toolkit.helpers import IntelligenceWorkflow
from toolkit.match_entity_records import prompts

from .detect import (
    build_attributes_dataframe,
    build_matches,
    build_matches_dataset,
    build_near_map,
    build_nearest_neighbors,
    build_sentence_pair_scores,
    convert_to_sentences,
)
from .prepare_model import (
    build_attribute_options,
    format_dataset,
)


class MatchEntityRecords(IntelligenceWorkflow):
    model_dfs: ClassVar[dict] = {}
    max_rows_to_process = 0
    attributes_list = []

    def __init__(self) -> None:
        pass

    @property
    def total_records(self):
        return sum(len(df) for df in self.model_dfs)

    @property
    def attribute_options(self) -> str:
        return build_attribute_options(self.model_dfs)

    def add_df_to_model(
        self,
        dataset: pl.DataFrame,
        entity_name_column: str,
        columns: list[str],
        dataset_name: str = "",
        id_column: str = "",
    ) -> pl.DataFrame:
        if not dataset_name:
            dataset_name = "dataset_" + len(self.model_dfs) + 1

        self.model_dfs[dataset_name] = format_dataset(
            dataset,
            columns,
            id_column,
            entity_name_column,
            self.max_rows_to_process,
        )
        return self.model_dfs[dataset_name]

    def build_model_df(self):
        self.model_df = build_attributes_dataframe(self.model_dfs, self.attributes_list)
        self.model_df = self.model_df.with_columns(
            (pl.col("Entity ID").cast(pl.Utf8))
            + "::"
            + pl.col("Dataset").alias("Unique ID")
        )

        self.sentences_vector_data = convert_to_sentences(self.model_df)
        return self.model_df

    async def embed_sentences(self):
        sentences_data = await OpenAIEmbedder(self.ai_configuration).embed_store_many(
            self.sentences_vector_data
        )
        texts = [x["text"] for x in sentences_data]
        self.embeddings = [
            np.array(next(d["vector"] for d in sentences_data if d["text"] == f))
            for f in texts
        ]

    def detect_record_groups(self, pair_embedding_threshold, pair_jaccard_threshold):
        distances, indices = build_nearest_neighbors(self.embeddings)
        near_map = build_near_map(
            distances,
            indices,
            self.sentences_vector_data,
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

        return build_matches_dataset(matches_df, pair_to_match, entity_to_group)

    def evaluate_groups(
        self,
        ai_instructions=prompts.user_prompt,
        callbacks: list | None = None,
    ) -> None:
        data = self.model_df.drop(
            [
                "Entity ID",
                "Dataset",
                "Name similarity",
            ]
        ).to_pandas()

        messages = utils.generate_batch_messages(
            ai_instructions, batch_name="data", batch_value=data
        )
        return OpenAIClient(self.ai_configuration).generate_chat(
            messages, callbacks=callbacks or []
        )

    def clear_model_dfs(self) -> None:
        self.model_dfs = {}
