# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl

import toolkit.match_entity_records.prompts as prompts
from toolkit.AI.defaults import DEFAULT_REPORT_BATCH_SIZE
from toolkit.AI.utils import generate_batch_messages


def prepare_for_ai_report(
    matches_df: pl.DataFrame,
    user_prompts: str | None = None,
    batch_size: int | None = DEFAULT_REPORT_BATCH_SIZE,
) -> list[dict[str, str]]:
    matches_df = matches_df.drop(
        [
            "Entity ID",
            "Dataset",
            "Name similarity",
        ]
    ).to_pandas()
    user_prompts = user_prompts or prompts.list_prompts

    return generate_batch_messages(
        user_prompts, batch_name="data", batch_value=matches_df, batch_size=batch_size
    )
