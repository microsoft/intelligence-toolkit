# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import asyncio
import python.AI.utils as utils
import python.question_answering.prompts as prompts
import python.question_answering.helper_functions as helper_functions
from json import loads

def assess_relevance(
        ai_configuration,
        search_label,
        search_chunks,
        question,
        logit_bias,
        relevance_test_limit,
        relevance_test_batch_size,
        test_history,
        progress_callback,
        chunk_callback
    ):
    print(f'Assessing relevance for {search_label} with {len(search_chunks)} chunks')
    batched_chunks = [search_chunks[i:i+relevance_test_batch_size]
                      for i in range(0, len(search_chunks), relevance_test_batch_size)]
    batched_messages = [[utils.prepare_messages(prompts.chunk_relevance_prompt, {'chunk': chunk, 'question': question}) 
                        for chunk in batch] for batch in batched_chunks]
    for mx, mapped_messages in enumerate(batched_messages):
        batch = batched_chunks[mx]
        if len(test_history) + len(mapped_messages) > relevance_test_limit:
            mapped_messages = mapped_messages[:relevance_test_limit - len(test_history)]
        mapped_responses = asyncio.run(helper_functions.map_generate_text(
            ai_configuration, mapped_messages, logit_bias=logit_bias, max_tokens=1))
        break_now = process_relevance_responses(
            search_label,
            batch,
            mapped_responses,
            test_history,
            progress_callback,
            chunk_callback
        )
        if break_now: # No relevant chunks found in this batch; terminate early
            break

def process_relevance_responses(
        search_label,
        search_chunks,
        mapped_responses,
        test_history,
        progress_callback,
        chunk_callback
    ):
    break_now = True
    for r, c in zip(mapped_responses, search_chunks):
        if c not in [x[1] for x in test_history]:
            test_history.append((search_label, c, r))
            if r == 'Yes':
                break_now = False
    if progress_callback is not None:
        progress_callback(helper_functions.get_test_progress(test_history))
    relevant_list = [x[1] for x in test_history if x[2] == 'Yes']
    if chunk_callback is not None:
        chunk_callback(relevant_list)
    return break_now

