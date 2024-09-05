# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import json

from toolkit.AI.base_embedder import BaseEmbedder
from toolkit.AI.classes import VectorData
from toolkit.AI.client import OpenAIClient
from toolkit.AI.utils import hash_text


def generate_text(ai_configuration, messages, **kwargs):
    return OpenAIClient(ai_configuration).generate_chat(messages, stream=False, **kwargs)

async def map_generate_text(ai_configuration, messages_list, **kwargs):
    return await OpenAIClient(ai_configuration).map_generate_text(
        messages_list, **kwargs
    )


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
    print(test_history[-3:])
    print(response)
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

async def embed_texts(
    cid_to_text, text_embedder: BaseEmbedder, cache_data=True, callbacks=[]
) -> dict:
    cid_to_vector = {}
    data: list[VectorData] = []

    for cid, text in cid_to_text.items():
        data.append(
            {"hash": hash_text(text), "text": text, "additional_details": {"cid": cid}}
        )

    embedded_data = await text_embedder.embed_store_many(data, callbacks, cache_data)
    for item in embedded_data:
        details = json.loads(item["additional_details"])
        if len(details.keys()) == 0:
            continue
        cid_to_vector[details["cid"]] = item["vector"]
    return cid_to_vector

