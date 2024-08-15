# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict
import asyncio
import pdfkit
import pdfplumber
import tempfile
import io
from textblob import TextBlob
import networkx as nx
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

def generate_answer(
        process_chunks,
        sorted_chunks,
        answer_batch_size,
        question,
        chunk_to_metadata,
        progress_stream,
        progress_callback,
        answer_stream,
        answer_callback
        ):
    selected_chunks = process_chunks[:answer_batch_size]
    formatted_chunks = [f'{chunk_to_metadata[c]}:\n\n{c}\n\n' for c in selected_chunks]
    answer_messages = utils.prepare_messages(prompts.chunk_summarization_prompt, {'chunks': formatted_chunks, 'question': question})
    answer_response = ui_components.generate_text(answer_messages)
    answer_stream = [answer_response] + answer_stream
    answer_callback(answer_stream)
    return selected_chunks, progress_stream, answer_stream

def check_generate_answer(
        process_chunks,
        sorted_chunks,
        use_all,
        answer_batch_size,
        question,
        chunk_to_metadata,
        progress_stream,
        progress_callback,
        answer_stream,
        answer_callback
        ):
    all_selected_chunks = []
    if use_all:
        remaining_chunks = set(process_chunks) - set(all_selected_chunks)
        while len(remaining_chunks) > 0:
            selected_chunks, progress_stream, answer_stream = generate_answer(
                remaining_chunks,
                sorted_chunks,
                answer_batch_size,
                question, chunk_to_metadata,
                progress_stream,
                progress_callback,
                answer_stream,
                answer_callback
            )
            all_selected_chunks.extend(selected_chunks)
            remaining_chunks = set(process_chunks) - set(all_selected_chunks)
    if len(process_chunks) >= answer_batch_size:
        selected_chunks, progress_stream, answer_stream = generate_answer(
            process_chunks,
            sorted_chunks,
            answer_batch_size,
            question, chunk_to_metadata,
            progress_stream,
            progress_callback,
            answer_stream,
            answer_callback
        )
        all_selected_chunks.extend(selected_chunks)
    return all_selected_chunks, progress_stream, answer_stream

def search_answers(
        question,
        text_to_chunks,
        chunk_to_concepts,
        concept_to_chunks,
        text_to_vectors,
        concept_graph,
        community_to_concepts,
        concept_to_community,
        previous_chunk,
        next_chunk,
        embedder,
        embedding_cache,
        max_outer_iterations,
        relevance_batch_size,
        answer_batch_size,
        select_logit_bias,
        chunk_successive_no_break,
        community_successive_no_break,
        community_chunk_successive_no_break,
        progress_callback,
        answer_callback,
    ):
    progress_stream = []
    answer_stream = []
    chunk_to_metadata = {}
    chunk_status = defaultdict(lambda: ChunkStatus.UNSEEN)
    processing_queue = []
    sorted_chunks = []
    for text, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
            chunk_to_metadata[chunk] = f'File: {text}; Chunk: {cx+1})'
            sorted_chunks.append(chunk)
    sorted_chunks.sort()
    all_units = []
    for text, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
            all_units.append(
                (
                    text,
                    chunk,
                    text_to_vectors[text][cx]
                )
            )
    scratchpad = ''
    primary_results = []
    yes_id = tiktoken.get_encoding('o200k_base').encode('Yes')[0]
    no_id = tiktoken.get_encoding('o200k_base').encode('No')[0]
    logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}
    for i in range(max_outer_iterations):
        augmented_question = f'Question: {question} {scratchpad}'
        aq_embedding = np.array(
            embedder.embed_store_one(
                augmented_question, embedding_cache
            )
        )
        cosine_distances = sorted(
            [
                (t, c, scipy.spatial.distance.cosine(aq_embedding, v))
                for (t, c, v) in all_units if chunk_status[c] in [ChunkStatus.UNSEEN]
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
            if r == 'Yes':
                chunk_status[c] = ChunkStatus.RELEVANT_UNPROCESSED
            else:
                chunk_status[c] = ChunkStatus.IRRELEVANT
        yes_chunks = [c for c in closest_batch if chunk_status[c] == ChunkStatus.RELEVANT_UNPROCESSED]
        primary_response = "Yes" if len(yes_chunks) > 0 else "No"
        yes_id = sorted_chunks.index(yes_chunks[0]) if len(yes_chunks) > 0 else '-'
        primary_results.append(primary_response)

        if primary_response == 'Yes':
            closest = yes_chunks[0]
            if closest not in processing_queue and chunk_status[closest] != ChunkStatus.RELEVANT_PROCESSED:
                processing_queue.append(closest)
                progress_stream = [closest] + progress_stream
                progress_callback(progress_stream)
            adjacent = get_adjacent_chunks(closest, previous_chunk, next_chunk)
            for adj in adjacent:
                if chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                    chunk_status[adj] = ChunkStatus.RELEVANT_UNPROCESSED
                    if adj not in processing_queue and chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                        processing_queue.append(adj)
                        progress_stream = [adj] + progress_stream
                        progress_callback(progress_stream)
            concepts = chunk_to_concepts[closest]
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
            secondary_results = []
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
                        if r == 'Yes':
                            chunk_status[c] = ChunkStatus.RELEVANT_UNPROCESSED
                            if c not in processing_queue and chunk_status[c] != ChunkStatus.RELEVANT_PROCESSED:
                                processing_queue.append(c)
                                progress_stream = [c] + progress_stream
                                progress_callback(progress_stream)
                            adjacent = get_adjacent_chunks(closest, previous_chunk, next_chunk)
                            for adj in adjacent:
                                if chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                                    chunk_status[adj] = ChunkStatus.RELEVANT_UNPROCESSED
                                    if adj not in processing_queue and chunk_status[adj] != ChunkStatus.RELEVANT_PROCESSED:
                                        processing_queue.append(adj)
                                        progress_stream = [adj] + progress_stream
                                        progress_callback(progress_stream)
                        else:
                            chunk_status[c] = ChunkStatus.IRRELEVANT

                    last_results = tertiary_results[-community_chunk_successive_no_break:]
                    if len(last_results) == community_chunk_successive_no_break and all([r == 'No' for r in last_results]):
                        processed_chunks, progress_stream, answer_stream = check_generate_answer(
                            process_chunks=processing_queue,
                            sorted_chunks=sorted_chunks,
                            use_all=False,
                            answer_batch_size=answer_batch_size,
                            question=question,
                            chunk_to_metadata=chunk_to_metadata,
                            progress_stream=progress_stream,
                            progress_callback=progress_callback,
                            answer_stream=answer_stream,
                            answer_callback=answer_callback
                        )
                        for c in processed_chunks:
                            chunk_status[c] = ChunkStatus.RELEVANT_PROCESSED
                            processing_queue.remove(c)
                        break
                        
                if len(tertiary_results) > community_chunk_successive_no_break: # must have had at least one relevant chunk
                    secondary_results.append('Yes')
                else:
                    secondary_results.append('No')
                    last_results = secondary_results[-community_successive_no_break:]
                    if len(last_results) == community_successive_no_break and all([r == 'No' for r in last_results]):
                        break
            processed_chunks, progress_stream, answer_stream = check_generate_answer(
                process_chunks=processing_queue,
                sorted_chunks=sorted_chunks,
                use_all=False,
                answer_batch_size=answer_batch_size,
                question=question,
                chunk_to_metadata=chunk_to_metadata,
                progress_stream=progress_stream,
                progress_callback=progress_callback,
                answer_stream=answer_stream,
                answer_callback=answer_callback
            )
            for c in processed_chunks:
                chunk_status[c] = ChunkStatus.RELEVANT_PROCESSED
                processing_queue.remove(c)
        else:
            last_results = primary_results[-chunk_successive_no_break:]
            if len(last_results) == chunk_successive_no_break and all([r == '--No' for r in last_results]):
                break
    processed_chunks, progress_stream, answer_stream = check_generate_answer(
        process_chunks=processing_queue,
        sorted_chunks=sorted_chunks,
        use_all=True,
        answer_batch_size=answer_batch_size,
        question=question,
        chunk_to_metadata=chunk_to_metadata,
        progress_stream=progress_stream,
        progress_callback=progress_callback,
        answer_stream=answer_stream,
        answer_callback=answer_callback
    )
    for c in processed_chunks:
        chunk_status[c] = ChunkStatus.RELEVANT_PROCESSED
        processing_queue.remove(c)
    return progress_stream, answer_stream