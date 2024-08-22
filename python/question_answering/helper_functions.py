# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from python.AI.client import OpenAIClient

def generate_text(ai_configuration, messages, **kwargs):
    return OpenAIClient(ai_configuration).generate_chat(messages, stream=False, **kwargs)

def map_generate_text(ai_configuration, messages_list, **kwargs):
    return OpenAIClient(ai_configuration).map_generate_text(messages_list, **kwargs)

def get_adjacent_chunks(source, previous_chunk_dict, next_chunk_dict, steps):
    prev_chunks = []
    current_chunk = source
    for i in range(steps):
        prev_chunk = previous_chunk_dict.get(current_chunk, None)
        if prev_chunk is None:
            break
        prev_chunks.append(prev_chunk)
        current_chunk = prev_chunk
    next_chunks = []
    current_chunk = source
    for i in range(steps):
        next_chunk = next_chunk_dict.get(current_chunk, None)
        if next_chunk is None:
            break
        next_chunks.append(next_chunk)
        current_chunk = next_chunk
    return set(prev_chunks + next_chunks)

def get_test_progress(test_history):
    current_search = ''
    current_relevant = 0
    current_tested = 0
    total_relevant = 0
    total_tested = 0
    rounds = []
    for ix, (search, chunk, response) in enumerate(test_history):
        if search != current_search:
            if current_search != '':
                rounds.append(f'{current_search}: {current_relevant}/{current_tested}')
            current_search = search
            current_relevant = 0
            current_tested = 0
        current_tested += 1
        total_tested += 1
        if response == 'Yes':
            current_relevant += 1
            total_relevant += 1
    if current_search != '':
        rounds.append(f'{current_search}: {current_relevant}/{current_tested}')
    response = f'**Chunks relevant/tested: {total_relevant}/{total_tested}**'
    if len(rounds) > 0:
        response += ' (' + '; '.join(rounds) + ')'
    return response

def get_answer_progress(answer_history):
    if len(answer_history) == 0:
        return ''
    sum_used = sum([x[0] for x in answer_history])
    sum_provided = sum([x[1] for x in answer_history])
    progress = f'**Chunks referenced/relevant: {sum_used}/{sum_provided}**'
    if len(answer_history) > 0:
        progress += ' ('
        for used_chunks, provided_chunks in answer_history:
            progress += f'{used_chunks}/{provided_chunks}, '
        progress = progress[:-2] + ')'
    return progress

def test_history_elements(test_history):
    relevant_list = [x[1] for x in test_history if x[2] == 'Yes']
    seen_list = [x[1] for x in test_history]
    return relevant_list, seen_list
