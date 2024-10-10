# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import pandas as pd

import toolkit.AI.utils as utils
import toolkit.generate_mock_data.prompts as prompts
from toolkit.helpers.progress_batch_callback import ProgressBatchCallback


async def generate_text_data(
    ai_configuration,
    generation_guidance,
    input_texts,
    temperature,
    parallel_threads,
    df_update_callback,
    callback_batch,
):
    generated_texts = []
    
    batched_texts = [input_texts[i:i + parallel_threads] for i in range(0, len(input_texts), parallel_threads)]

    df = pd.DataFrame(columns=["mock_text"])

    for batch in batched_texts:
        new_texts = await _generate_text_parallel(
            ai_configuration=ai_configuration,
            input_texts=batch,
            generation_guidance=generation_guidance,
            temperature=temperature,
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
    temperature,
    callbacks: list[ProgressBatchCallback] | None = None,
):
    mapped_messages = [utils.prepare_messages(
        prompts.text_generation_prompt, 
        {
            'input_text': input_text,
            'generation_guidance': generation_guidance,
        }) for input_text in input_texts
    ]

    return await utils.map_generate_text(
        ai_configuration,
        mapped_messages,
        temperature=temperature,
        callbacks=callbacks,
    )

