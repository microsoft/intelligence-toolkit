# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import asyncio
from json import loads

import toolkit.AI.utils as utils
import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.prompts as prompts


async def assess_relevance(
    ai_configuration,
    search_label,
    search_cids,
    cid_to_text,
    question,
    logit_bias,
    relevance_test_budget,
    relevance_test_batch_size,
    test_history,
    progress_callback,
    chunk_callback,
):
    print(f'Assessing relevance for {search_label} with {len(search_cids)} chunks')
    batched_cids = [search_cids[i:i+relevance_test_batch_size]
                      for i in range(0, len(search_cids), relevance_test_batch_size)]
    batched_texts = [[cid_to_text[cid] for cid in batch] for batch in batched_cids]
    batched_messages = [[utils.prepare_messages(prompts.chunk_relevance_prompt, {'chunk': chunk, 'question': question}) 
                        for chunk in batch] for batch in batched_texts]
    is_relevant = False
    for mx, mapped_messages in enumerate(batched_messages):
        cid_batch = batched_cids[mx]
        if len(test_history) + len(mapped_messages) > relevance_test_budget:
            mapped_messages = mapped_messages[:relevance_test_budget - len(test_history)]
        mapped_responses = await helper_functions.map_generate_text(
            ai_configuration, mapped_messages, logit_bias=logit_bias, max_tokens=1
        )
        num_relevant = process_relevance_responses(
            search_label,
            cid_batch,
            cid_to_text,
            mapped_responses,
            test_history,
            progress_callback,
            chunk_callback
        )
        print(f'Batch {mx+1} of {len(batched_messages)}: {num_relevant} relevant chunks')
        is_relevant = num_relevant > 0
        if not is_relevant: # No relevant chunks found in this batch; terminate early
            break
    return is_relevant

def process_relevance_responses(
        search_label,
        search_cids,
        cid_to_text,
        mapped_responses,
        test_history,
        progress_callback,
        chunk_callback
    ):
    num_relevant = 0
    for r, c in zip(mapped_responses, search_cids):
        if c not in [x[1] for x in test_history]:
            test_history.append((search_label, c, r))
            if r == 'Yes':
                num_relevant += 1
    if progress_callback is not None:
        progress_callback(helper_functions.get_test_progress(test_history))
    relevant_list = [x[1] for x in test_history if x[2] == 'Yes']
    if chunk_callback is not None:
        chunk_callback([cid_to_text[cid] for cid in relevant_list])
    return num_relevant

