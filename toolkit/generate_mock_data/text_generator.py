# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import toolkit.AI.utils as utils
import toolkit.generate_mock_data.prompts as prompts
from toolkit.AI.openai_configuration import OpenAIConfiguration


async def generate_text_data(
    ai_configuration: OpenAIConfiguration,
    input_texts: list[str],
    generation_guidance: str="",
    temperature: float=0.5,
    df_update_callback=None,
):
    df = pd.DataFrame(columns=["mock_text"])
    tasks = []
    for text in input_texts:
        tasks.append(asyncio.create_task(_generate_text_async(
            ai_configuration=ai_configuration,
            input_text=text,
            generation_guidance=generation_guidance,
            temperature=temperature
        )))
    generated_texts = await tqdm_asyncio.gather(*tasks)
    df = pd.DataFrame(generated_texts, columns=["mock_text"])
    if df_update_callback is not None:
        df_update_callback(df)

    return generated_texts, df


async def _generate_text_async(
    ai_configuration,
    input_text,
    generation_guidance,
    temperature
):
    messages = utils.prepare_messages(
        prompts.text_generation_prompt, 
        {
            'input_text': input_text,
            'generation_guidance': generation_guidance,
        }
    )

    return await utils.generate_text_async(
        ai_configuration,
        messages,
        temperature=temperature
    )

