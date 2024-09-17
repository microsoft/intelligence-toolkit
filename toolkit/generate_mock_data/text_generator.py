# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import asyncio
import random
import re
from json import dumps, loads
from operator import call

import pandas as pd

import toolkit.AI.utils as utils
import toolkit.generate_mock_data.prompts as prompts
import toolkit.query_text_data.helper_functions as helper_functions
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback


async def generate_text_data(
    ai_configuration,
    generation_guidance,
    input_texts,
    df_update_callback,
    callback_batch,
):
    generated_texts = []
    
    new_texts = await _generate_text_parallel(
        ai_configuration=ai_configuration,
        input_texts=input_texts,
        generation_guidance=generation_guidance,
        callbacks=[callback_batch] if callback_batch is not None else None,
    )
    generated_texts.extend(new_texts)

    df = pd.DataFrame(generated_texts, columns=["mock_text"])
    if df_update_callback is not None:
        df_update_callback(df)
    return generated_texts, df


async def _generate_text_parallel(
    ai_configuration,
    input_texts,
    generation_guidance,
    callbacks: list[ProgressBatchCallback] | None = None,
):
    mapped_messages = [utils.prepare_messages(
        prompts.text_generation_prompt, 
        {
            'input_text': input_text,
            'generation_guidance': generation_guidance,
        }) for input_text in input_texts
    ]

    return await helper_functions.map_generate_text(
        ai_configuration,
        mapped_messages,
        temperature=0.75,
        callbacks=callbacks,
    )

