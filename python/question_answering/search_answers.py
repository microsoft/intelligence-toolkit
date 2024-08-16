# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict, Counter
import asyncio
import scipy.spatial.distance
import app.util.ui_components as ui_components
import python.AI.utils as utils
import python.question_answering.prompts as prompts
import tiktoken
from enum import Enum

class ChunkStatus(Enum):
    UNSEEN = 0
    IRRELEVANT = 1
    RELEVANT_UNPROCESSED = 2
    RELEVANT_PROCESSED = 3

def get_adjacent_chunks(closest, previous_chunk, next_chunk):
    prev_chunk = previous_chunk.get(closest, None)
    next_chunk = next_chunk.get(closest, None)
    return [x for x in [prev_chunk, next_chunk] if x is not None]

def should_switch(test_history, switch_on_successive_irrelevant):
    last_results = test_history[-switch_on_successive_irrelevant:]
    return len(last_results) == switch_on_successive_irrelevant and all([r == 'No' for (c, r) in last_results])

def should_terminate(test_history, terminate_on_chunks_tested, terminate_on_relevant_chunks, terminate_on_successive_irrelevant):
    should_terminate = False
    if terminate_on_chunks_tested != 0 and len(test_history) >= terminate_on_chunks_tested:
        should_terminate = True
    if terminate_on_relevant_chunks != 0 and len([x for x in test_history if x[1] == 'Yes']) >= terminate_on_relevant_chunks:
        should_terminate = True
    if terminate_on_successive_irrelevant != 0:
        last_results = [x[1] for x in test_history[-terminate_on_successive_irrelevant:]]
        if len(last_results) == terminate_on_successive_irrelevant and all([r == 'No' for r in last_results]):
            should_terminate = True
    return should_terminate

def should_switch(test_history, switch_on_successive_irrelevant):
    last_results = test_history[-switch_on_successive_irrelevant:]
    return len(last_results) == switch_on_successive_irrelevant and all([r == 'No' for (c, r) in last_results])

def get_progress(test_history):
    chunks_tested = len(test_history)
    relevant_chunks = len([x for x in test_history if x[1] == 'Yes'])
    successive_irrelevant_list = list(reversed([x[1] for x in test_history]))
    successive_irrelevant = successive_irrelevant_list.index('Yes') if 'Yes' in successive_irrelevant_list else len(successive_irrelevant_list)
    response = {
        'chunks_tested': chunks_tested,
        'relevant_chunks': relevant_chunks,
        'successive_irrelevant': successive_irrelevant
    }
    return response

def generate_answer(
        process_chunks,
        answer_batch_size,
        question,
        chunk_to_metadata,
        answer_stream,
        answer_callback
        ):
    selected_chunks = process_chunks[:answer_batch_size]
    formatted_chunks = [f'{chunk_to_metadata[c]}:\n\n{c}\n\n' for c in selected_chunks]
    answer_messages = utils.prepare_messages(prompts.chunk_summarization_prompt, {'chunks': formatted_chunks, 'question': question})
    answer_response = ui_components.generate_text(answer_messages)
    answer_stream = [answer_response] + answer_stream
    answer_callback(answer_stream)
    return selected_chunks, answer_stream

def check_generate_answer(
        process_chunks,
        use_all,
        answer_batch_size,
        question,
        chunk_to_metadata,
        answer_stream,
        answer_callback
        ):
    all_selected_chunks = []
    if use_all:
        remaining_chunks = list(set(process_chunks) - set(all_selected_chunks))
        while len(remaining_chunks) > 0:
            selected_chunks, answer_stream = generate_answer(
                remaining_chunks,
                answer_batch_size,
                question, chunk_to_metadata,
                answer_stream,
                answer_callback
            )
            all_selected_chunks.extend(selected_chunks)
            remaining_chunks = list(set(process_chunks) - set(all_selected_chunks))
    if len(process_chunks) >= answer_batch_size:
        selected_chunks, answer_stream = generate_answer(
            process_chunks,
            answer_batch_size,
            question, chunk_to_metadata,
            answer_stream,
            answer_callback
        )
        all_selected_chunks.extend(selected_chunks)
    return all_selected_chunks, answer_stream

def search_answers(
        question,
        text_to_chunks,
        chunk_to_concepts,
        concept_to_chunks,
        text_to_vectors,
        community_to_concepts,
        concept_to_community,
        previous_chunk,
        next_chunk,
        embedder,
        embedding_cache,
        relevance_batch_size,
        answer_batch_size,
        select_logit_bias,
        terminate_on_chunks_tested,
        terminate_on_relevant_chunks,
        terminate_on_successive_irrelevant,
        switch_on_successive_irrelevant,
        augment_top_concepts,
        progress_callback,
        chunk_callback,
        answer_callback,
    ):
    progress_stream = []
    answer_stream = []
    chunk_to_metadata = {}
    chunk_status = defaultdict(lambda: ChunkStatus.UNSEEN)
    processing_queue = []
    sorted_chunks = []
    test_history = []
    for text, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
            chunk_to_metadata[chunk] = f'File: {text}; Chunk: {cx+1})'
            sorted_chunks.append(chunk)
    all_units = []
    all_matched_concepts = Counter()
    for text, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
            all_units.append(
                (
                    text,
                    chunk,
                    text_to_vectors[text][cx]
                )
            )

    yes_id = tiktoken.get_encoding('o200k_base').encode('Yes')[0]
    no_id = tiktoken.get_encoding('o200k_base').encode('No')[0]
    logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}

    progress_callback(get_progress(test_history))
    while not should_terminate(test_history, terminate_on_chunks_tested, terminate_on_relevant_chunks, terminate_on_successive_irrelevant):
        augmented_question = f'Question: {question} Concepts: {"; ".join([x[0] for x in all_matched_concepts.most_common(augment_top_concepts)])}'
        print(augmented_question)
        aq_embedding = np.array(
            embedder.embed_store_one(
                augmented_question, embedding_cache
            )
        )
        cosine_distances = sorted(
            [
                (t, c, scipy.spatial.distance.cosine(aq_embedding, v))
                for (t, c, v) in all_units if chunk_status[c] == ChunkStatus.UNSEEN
            ],
            key=lambda x: x[2],
            reverse=False,
        )
        if len(cosine_distances) == 0:
            break
        closest_batch = [x[1] for x in cosine_distances[:relevance_batch_size]]
        primary_mapped_messages = [utils.prepare_messages(prompts.chunk_relevance_prompt, {'chunk': chunk, 'question': question}) 
                                   for chunk in closest_batch]
        primary_mapped_responses = asyncio.run(ui_components.map_generate_text(primary_mapped_messages, logit_bias=logit_bias, max_tokens=1))
        for r, c in zip(primary_mapped_responses, closest_batch):
            if c not in [x[0] for x in test_history]:
                test_history.append((c, r))
            if r == 'Yes':
                chunk_status[c] = ChunkStatus.RELEVANT_UNPROCESSED
            else:
                chunk_status[c] = ChunkStatus.IRRELEVANT
        progress_callback(get_progress(test_history))
        yes_chunks = [c for c in closest_batch if chunk_status[c] == ChunkStatus.RELEVANT_UNPROCESSED]
        primary_response = "Yes" if len(yes_chunks) > 0 else "No"
        yes_id = sorted_chunks.index(yes_chunks[0]) if len(yes_chunks) > 0 else '-'

        if primary_response == 'Yes':
            closest = yes_chunks[0]
            if closest not in processing_queue and chunk_status[closest] != ChunkStatus.RELEVANT_PROCESSED:
                processing_queue.append(closest)
                progress_stream = [closest] + progress_stream
                chunk_callback(progress_stream)
            adjacent = get_adjacent_chunks(closest, previous_chunk, next_chunk)
            for adj in adjacent:
                if chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                    chunk_status[adj] = ChunkStatus.RELEVANT_UNPROCESSED
                    if adj not in processing_queue and chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                        processing_queue.append(adj)
                        all_matched_concepts.update(chunk_to_concepts[adj])
                        progress_stream = [adj] + progress_stream
                        chunk_callback(progress_stream)
            concepts = chunk_to_concepts[closest]
            all_matched_concepts.update(concepts)
            community_concept_counts = defaultdict(int)
            for concept in concepts:
                if concept not in concept_to_community.keys():
                    continue
                community = concept_to_community[concept]
                community_concept_counts[community] += 1
            sorted_communities = sorted(
                community_concept_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for community, community_concept_count in sorted_communities:
                linked_chunks = set()
                for concept in community_to_concepts[community]:
                    candidate_chunks = concept_to_chunks[concept]
                    filtered_chunks = [c for c in candidate_chunks if chunk_status[c] == ChunkStatus.UNSEEN]
                    linked_chunks.update(filtered_chunks)
                chunk_concept_matches = {}
                for chunk in linked_chunks:
                    chunk_concept_matches[chunk] = len(set(chunk_to_concepts[chunk]).intersection(concepts))
                sorted_chunk_concept_matches = sorted(
                    chunk_concept_matches.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                tertiary_results = []
                new_chunks = [chunk for chunk, _ in sorted_chunk_concept_matches if chunk_status[chunk] == ChunkStatus.UNSEEN]
                
                batches = [new_chunks[i:i+relevance_batch_size] for i in range(0, len(new_chunks), relevance_batch_size)]
                for batch in batches:
                    mapped_messages = [utils.prepare_messages(prompts.chunk_relevance_prompt, {'chunk': chunk, 'question': question})
                                       for chunk in batch]
                    mapped_responses = asyncio.run(ui_components.map_generate_text(mapped_messages, logit_bias=logit_bias, max_tokens=1))
                    for ri, mapped_response in enumerate(mapped_responses):
                        tertiary_results.append(mapped_response)
                    for r, c in zip(mapped_responses, batch):
                        if c not in [x[0] for x in test_history]:
                            test_history.append((c, r))
                        if r == 'Yes':
                            chunk_status[c] = ChunkStatus.RELEVANT_UNPROCESSED
                            if c not in processing_queue and chunk_status[c] != ChunkStatus.RELEVANT_PROCESSED:
                                processing_queue.append(c)
                                progress_stream = [c] + progress_stream
                                chunk_callback(progress_stream)
                                all_matched_concepts.update(chunk_to_concepts[c])
                            adjacent = get_adjacent_chunks(closest, previous_chunk, next_chunk)
                            for adj in adjacent:
                                if chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                                    chunk_status[adj] = ChunkStatus.RELEVANT_UNPROCESSED
                                    if adj not in processing_queue and chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                                        processing_queue.append(adj)
                                        progress_stream = [adj] + progress_stream
                                        chunk_callback(progress_stream)
                                        all_matched_concepts.update(chunk_to_concepts[adj])
                        else:
                            chunk_status[c] = ChunkStatus.IRRELEVANT
                    progress_callback(get_progress(test_history))
                    if should_terminate(test_history, terminate_on_chunks_tested, terminate_on_relevant_chunks, terminate_on_successive_irrelevant):
                        break
                    if should_switch(test_history, switch_on_successive_irrelevant):
                        processed_chunks, answer_stream = check_generate_answer(
                            process_chunks=processing_queue,
                            use_all=False,
                            answer_batch_size=answer_batch_size,
                            question=question,
                            chunk_to_metadata=chunk_to_metadata,
                            answer_stream=answer_stream,
                            answer_callback=answer_callback
                        )
                        for c in processed_chunks:
                            chunk_status[c] = ChunkStatus.RELEVANT_PROCESSED
                            processing_queue.remove(c)
                        break
                        
            processed_chunks, answer_stream = check_generate_answer(
                process_chunks=processing_queue,
                use_all=False,
                answer_batch_size=answer_batch_size,
                question=question,
                chunk_to_metadata=chunk_to_metadata,
                answer_stream=answer_stream,
                answer_callback=answer_callback
            )
            for c in processed_chunks:
                chunk_status[c] = ChunkStatus.RELEVANT_PROCESSED
                processing_queue.remove(c)
        progress_callback(get_progress(test_history))
    processed_chunks, answer_stream = check_generate_answer(
        process_chunks=processing_queue,
        sorted_chunks=sorted_chunks,
        use_all=True,
        answer_batch_size=answer_batch_size,
        question=question,
        chunk_to_metadata=chunk_to_metadata,
        answer_stream=answer_stream,
        answer_callback=answer_callback
    )
    for c in processed_chunks:
        chunk_status[c] = ChunkStatus.RELEVANT_PROCESSED
        processing_queue.remove(c)
    return progress_stream, answer_stream