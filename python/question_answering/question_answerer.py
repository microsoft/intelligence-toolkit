# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
from collections import defaultdict, Counter
import scipy.spatial.distance
import python.question_answering.answer_schema as answer_schema
import python.question_answering.relevance_assessor as relevance_assessor
import python.question_answering.answer_builder as answer_builder
import python.question_answering.helper_functions as helper_functions
import tiktoken

def answer_question(
        ai_configuration,
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
        answer_batch_size,
        select_logit_bias,
        semantic_search_depth,
        structural_search_steps,
        community_search_breadth,
        relevance_test_limit,
        relevance_test_batch_size,
        augment_top_concepts,
        chunk_progress_callback=None,
        answer_progress_callback=None,
        chunk_callback=None,
        answer_callback=None
    ):
    answer_format = answer_schema.answer_format
    answer_object = {
        "question": question,
        "title": "",
        "introduction": "",
        "content_id_sequence": [],
        "content_items": [],
        "conclusion": ""
    }
    answer_stream = []
    structural_processed = set()
    sorted_chunks = []
    test_history = []
    answer_history = []
    for text, chunks in text_to_chunks.items():
        for cx, chunk in enumerate(chunks):
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

    community_to_ranked_chunks = {}
    for community, concepts in community_to_concepts.items():
        community_chunks = set()
        for concept in concepts:
            candidate_chunks = concept_to_chunks[concept]
            filtered_chunks = [c for c in candidate_chunks if c not in community_chunks]
            community_chunks.update(filtered_chunks)
        community_to_ranked_chunks[community] = sorted(
            community_chunks,
            key=lambda x: len(set(chunk_to_concepts[x]).intersection(concepts)),
            reverse=True
        )

    yes_id = tiktoken.get_encoding('o200k_base').encode('Yes')[0]
    no_id = tiktoken.get_encoding('o200k_base').encode('No')[0]
    logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}
    if chunk_progress_callback is not None:
        chunk_progress_callback(helper_functions.get_test_progress(test_history))
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
        aq_embedding = np.array(
            embedder.embed_store_one(
                augmented_question, embedding_cache
            )
        )
        relevant, seen = helper_functions.test_history_elements(test_history)
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
        relevance_assessor.assess_relevance(
            ai_configuration=ai_configuration,
            search_label='depth',
            search_chunks=semantic_search_chunks,
            question=question,
            logit_bias=logit_bias,
            relevance_test_limit=relevance_test_limit,
            relevance_test_batch_size=relevance_test_batch_size,
            test_history=test_history,
            progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback,
        )
        
        # Next, we do breadth/community search, which is a search of the unseen chunks most representative of the communities
        # whose concepts are contained in current chunks
        current_concepts = set()
        relevant, seen = helper_functions.test_history_elements(test_history)
        for chunk in relevant:
            concepts = chunk_to_concepts[chunk]
            current_concepts.update(concepts)

        last_round_matched_concepts.update(current_concepts)
        community_concept_counts = {}
        for community in community_to_concepts.keys():
            community_concept_counts[community] = 0
        for concept in current_concepts:
            if concept not in concept_to_community.keys():
                continue
            community = concept_to_community[concept]
            community_concept_counts[community] += concept_graph.nodes[concept]['count']
        sorted_communities = sorted(
            community_concept_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        community_search_chunks = []
        can_continue = True
        while len(community_search_chunks) < community_search_breadth and can_continue:
            can_continue = False
            for community, _ in sorted_communities:
                unseen_community_chunks = community_to_ranked_chunks[community]
                unseen_community_chunks = [c for c in unseen_community_chunks if c not in seen and c not in community_search_chunks]
                if len(unseen_community_chunks) > 0:
                    community_search_chunks.append(unseen_community_chunks[0])
                if len(community_search_chunks) > 1:
                    can_continue = True
                if len(community_search_chunks) == community_search_breadth:
                    break
            
        relevance_assessor.assess_relevance(
            ai_configuration=ai_configuration,
            search_label='breadth',
            search_chunks=community_search_chunks,
            question=question,
            logit_bias=logit_bias,
            relevance_test_limit=relevance_test_limit,
            relevance_test_batch_size=relevance_test_batch_size,
            test_history=test_history,
            progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback,
        )

        # Finally, we do detail/structural search, which is a search of the chunks adjacent to the relevant chunks.
        relevant, seen = helper_functions.test_history_elements(test_history)
        structural_sources = [c for c in relevant if c not in structural_processed]
        structural_targets = set()
        for c in structural_sources:
            structural_targets.update(helper_functions.get_adjacent_chunks(c, previous_chunk, next_chunk, structural_search_steps))
        structural_search_chunks = [x for x in structural_targets if x not in seen]
        structural_processed.update(structural_search_chunks)

        relevance_assessor.assess_relevance(
            ai_configuration=ai_configuration,
            search_label='detail',
            search_chunks=structural_search_chunks,
            question=question,
            logit_bias=logit_bias,
            relevance_test_limit=relevance_test_limit,
            relevance_test_batch_size=relevance_test_batch_size,
            test_history=test_history,
            progress_callback=chunk_progress_callback,
            chunk_callback=chunk_callback,
        )

    relevant, seen = helper_functions.test_history_elements(test_history)
    relevant.sort(key=lambda x: sorted_chunks.index(x))

    answer_builder.generate_answers(
        ai_configuration=ai_configuration,
        answer_object=answer_object,
        answer_format=answer_format,
        process_chunks=relevant,
        answer_batch_size=answer_batch_size,
        answer_stream=answer_stream,
        answer_callback=answer_callback,
        answer_history=answer_history,
        progress_callback=answer_progress_callback,
    )
    return (
        relevant, 
        answer_stream,
        helper_functions.get_test_progress(test_history),
        helper_functions.get_answer_progress(answer_history)
    )