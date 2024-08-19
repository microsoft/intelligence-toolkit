# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict, Counter
import asyncio
import scipy.spatial.distance
import app.util.ui_components as ui_components
import python.AI.utils as utils
import python.question_answering.prompts as prompts
import tiktoken
import re
from json import dumps, loads

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

def extract_chunk_references(text):
    source_spans = re.finditer(r'\[source: (.+)\]', text, re.MULTILINE)
    references = set()
    for source_span in source_spans:
        parts = [x.strip() for x in source_span.group(1).split(',')]
        print(parts)
        references.update(parts)
    return references

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


def assess_relevance(
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
    batched_chunks = [search_chunks[i:i+relevance_test_batch_size] for i in range(0, len(search_chunks), relevance_test_batch_size)]
    batched_messages = [[utils.prepare_messages(prompts.chunk_relevance_prompt, {'chunk': chunk, 'question': question}) 
                                   for chunk in batch] for batch in batched_chunks]
    for mapped_messages in batched_messages:
        if len(test_history) + len(mapped_messages) > relevance_test_limit:
            mapped_messages = mapped_messages[:relevance_test_limit - len(test_history)]
        mapped_responses = asyncio.run(ui_components.map_generate_text(mapped_messages, logit_bias=logit_bias, max_tokens=1))
        break_now = process_relevance_responses(
            search_label,
            search_chunks,
            mapped_responses,
            test_history,
            progress_callback,
            chunk_callback
        )
        if break_now: # No relevant chunks found in this batch; terminate early
            break

def test_history_elements(test_history):
    relevant_list = [x[1] for x in test_history if x[2] == 'Yes']
    seen_list = [x[1] for x in test_history]
    return relevant_list, seen_list

def generate_answer(
        answer_object,
        answer_format,
        processing_queue,
        answer_batch_size,
        chunk_to_metadata,
        answer_stream,
        answer_callback
    ):
    selected_chunks = processing_queue[:answer_batch_size]
    for s in selected_chunks:
        processing_queue.remove(s)
    formatted_chunks = [f'{chunk_to_metadata[c]}:\n\n{c}\n\n'
                        for c in selected_chunks]
    answer_messages = utils.prepare_messages(
        prompts.chunk_summarization_prompt, 
        {'chunks': formatted_chunks, 'answer_object': answer_object}
    )
    answer_response = ui_components.generate_text(answer_messages, response_format=answer_format)
    update_answer_object(answer_object, loads(answer_response))
    answer_text = '\n\n'.join([x['content'] for x in answer_object['content_items']])
    references = extract_chunk_references(answer_text)
    answer_stream.insert(0, convert_answer_object_to_text(answer_object))
    answer_callback(answer_stream)
    return selected_chunks, references


def generate_answers(
        answer_object,
        answer_format,
        process_chunks,
        answer_batch_size,
        chunk_to_metadata,
        answer_stream,
        answer_callback,
        answer_history,
        progress_callback
    ):
    all_selected_chunks = []
    remaining_chunks = list(set(process_chunks) - set(all_selected_chunks))
    while len(remaining_chunks) > 0:
        selected_chunks, references = generate_answer(
            answer_object,
            answer_format,
            remaining_chunks,
            answer_batch_size,
            chunk_to_metadata,
            answer_stream,
            answer_callback
        )
        selected_metadata = set([chunk_to_metadata[c] for c in selected_chunks])
        used_references = [r for r in references if r in selected_metadata]
        answer_history.append((len(used_references), len(selected_chunks)))
        progress_callback(get_answer_progress(answer_history))
        all_selected_chunks.extend(selected_chunks)
        remaining_chunks = list(set(process_chunks) - set(all_selected_chunks))

def convert_answer_object_to_text(answer_object):
    response = f'# {answer_object["title"]}\n\n'
    response += f'*In response to: {answer_object["question"]}*\n\n'
    response += f'## Introduction\n\n{answer_object["introduction"]}\n\n## Analysis\n\n'
    for item in answer_object['content_items']:
        response += f'### {item["title"]}\n\n{item["content"]}\n\n'
    response += f'## Conclusion\n\n{answer_object["conclusion"]}\n\n'
    return response

def update_answer_object(answer_object, answer_update):
    new_and_updated_ids = set([x['id'] for x in answer_update['content_items']])
    answer_object['title'] = answer_update['title']
    answer_object['introduction'] = answer_update['introduction']
    answer_object['content_id_sequence'] = answer_update['content_id_sequence']
    print(answer_object['content_id_sequence'])
    updated_content_items = []
    for item_id in answer_object['content_id_sequence']:
        if item_id in new_and_updated_ids:
            for item in answer_update['content_items']:
                if item['id'] == item_id:
                    updated_content_items.append(item)
                    break
        else:
            for item in answer_object['content_items']:
                if item['id'] == item_id:
                    updated_content_items.append(item)
                    break
    answer_object['content_items'] = updated_content_items
    answer_object['conclusion'] = answer_update['conclusion']


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
    progress_callback(get_test_progress(test_history))
    relevant_list = [x[1] for x in test_history if x[2] == 'Yes']
    chunk_callback(relevant_list)
    return break_now

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
        answer_batch_size,
        select_logit_bias,
        semantic_search_depth,
        structural_search_steps,
        relational_search_depth,
        relevance_test_limit,
        relevance_test_batch_size,
        augment_top_concepts,
        chunk_progress_callback,
        answer_progress_callback,
        chunk_callback,
        answer_callback
    ):
    answer_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_object",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "title": {
                        "type": "string"
                    },
                    "introduction": {
                        "type": "string"
                    },
                    "content_id_sequence": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },
                    "content_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "number"
                                },
                                "title": {
                                    "type": "string"
                                },
                                "content": {
                                    "type": "string"
                                }
                            },
                            "required": ["id", "title", "content"],
                            "additionalProperties": False,
                        }
                    },
                    "conclusion": {
                        "type": "string"
                    },
                },
                "required": ["question", "title", "introduction", "content_id_sequence", "content_items", "conclusion"],
                "additionalProperties": False,
            }
        }
    }
    answer_object = {
        "question": question,
        "title": "",
        "introduction": "",
        "content_id_sequence": [],
        "content_items": [],
        "conclusion": ""
    }
    answer_stream = []
    chunk_to_metadata = {}
    structural_processed = set()
    sorted_chunks = []
    test_history = []
    answer_history = []
    for text, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
            chunk_to_metadata[chunk] = f'{text} ({cx+1})'
            sorted_chunks.append(chunk)
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

    yes_id = tiktoken.get_encoding('o200k_base').encode('Yes')[0]
    no_id = tiktoken.get_encoding('o200k_base').encode('No')[0]
    logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}
    chunk_progress_callback(get_test_progress(test_history))
    last_round_matched_concepts = Counter()

    # We repeat the query matching process for a fixed number of iterations, each time with a potentially different augmented question.
    # This allows information gathered from prior cycles to steer the search process into different parts of the semantic space.
    # Potentially useful for multi-hop queries.
    while len(test_history) < relevance_test_limit:
        
        augmented_question = f'Question: {question}'
        # We augment the question with the top concepts matched in the previous iterations.
        if len(last_round_matched_concepts) > 0:
            augmented_question += f'; Matched Concepts: {"; ".join([x[0] for x in last_round_matched_concepts.most_common(augment_top_concepts)])}'
            last_round_matched_concepts = Counter()
        print(augmented_question)
        aq_embedding = np.array(
            embedder.embed_store_one(
                augmented_question, embedding_cache
            )
        )
        relevant, seen = test_history_elements(test_history)
        cosine_distances = sorted(
            [
                (t, c, scipy.spatial.distance.cosine(aq_embedding, v))
                for (t, c, v) in all_units if c not in seen
            ],
            key=lambda x: x[2],
            reverse=False,
        )
        # We first do semantic search, analyze the matching chunks for relevance, and .
        semantic_search_chunks = [x[1] for x in cosine_distances[:semantic_search_depth]]
        assess_relevance(
            search_label='semantic',
            search_chunks=semantic_search_chunks,
            question=question,
            logit_bias=logit_bias,
            relevance_test_limit=relevance_test_limit,
            relevance_test_batch_size=relevance_test_batch_size,
            test_history=test_history,
            progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback,
        )
        # Next, we do structural search, which is a search of the chunks adjacent to the relevant chunks.
        relevant, seen = test_history_elements(test_history)
        structural_sources = [c for c in relevant if c not in structural_processed]
        structural_targets = set()
        for c in structural_sources:
            structural_targets.update(get_adjacent_chunks(c, previous_chunk, next_chunk, structural_search_steps))
        structural_search_chunks = [x for x in structural_targets if x not in seen]
        structural_processed.update(structural_search_chunks)

        assess_relevance(
            search_label='structural',
            search_chunks=structural_search_chunks,
            question=question,
            logit_bias=logit_bias,
            relevance_test_limit=relevance_test_limit,
            relevance_test_batch_size=relevance_test_batch_size,
            test_history=test_history,
            progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback,
        )
        # Finally, we do relational search, which is a search of the chunks linked to the relevant chunks by shared concepts.
        current_concepts = set()
        relevant, seen = test_history_elements(test_history)
        for chunk in relevant:
            concepts = chunk_to_concepts[chunk]
            current_concepts.update(concepts)

        last_round_matched_concepts.update(current_concepts)
        community_concept_counts = defaultdict(int)
        for concept in current_concepts:
            if concept not in concept_to_community.keys():
                continue
            community = concept_to_community[concept]
            community_concept_counts[community] += 1
        sorted_communities = sorted(
            community_concept_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        chunk_concept_matches = {}
        for community, _ in sorted_communities:
            filtered_community_concepts = set(community_to_concepts[community]).intersection(current_concepts)
            community_chunks = set()
            for concept in filtered_community_concepts:
                candidate_chunks = concept_to_chunks[concept]
                filtered_chunks = [c for c in candidate_chunks if c not in seen]
                community_chunks.update(filtered_chunks)
            
            for chunk in community_chunks:
                chunk_concept_matches[chunk] = len(set(chunk_to_concepts[chunk]).intersection(current_concepts))
        sorted_chunk_concept_matches = sorted(
            chunk_concept_matches.items(),
            key=lambda x: x[1],
            reverse=True
        )
        relational_search_chunks = [chunk for chunk, _ in sorted_chunk_concept_matches if chunk not in seen][:relational_search_depth]

        assess_relevance(
            search_label='relational',
            search_chunks=relational_search_chunks,
            question=question,
            logit_bias=logit_bias,
            relevance_test_limit=relevance_test_limit,
            relevance_test_batch_size=relevance_test_batch_size,
            test_history=test_history,
            progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback,
        )
    relevant, seen = test_history_elements(test_history)
    relevant.sort(key=lambda x: sorted_chunks.index(x))
    print(len(relevant))
    print([sorted_chunks.index(x) for x in relevant])
    generate_answers(
        answer_object=answer_object,
        answer_format=answer_format,
        process_chunks=relevant,
        answer_batch_size=answer_batch_size,
        chunk_to_metadata=chunk_to_metadata,
        answer_stream=answer_stream,
        answer_callback=answer_callback,
        answer_history=answer_history,
        progress_callback=answer_progress_callback,
    )
    return relevant, answer_stream, get_test_progress(test_history), get_answer_progress(answer_history)