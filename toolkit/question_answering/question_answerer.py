# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from collections import Counter, defaultdict

import numpy as np
import scipy.spatial.distance
import tiktoken

import toolkit.question_answering.answer_builder as answer_builder
import toolkit.question_answering.answer_schema as answer_schema
import toolkit.question_answering.helper_functions as helper_functions
import toolkit.question_answering.relevance_assessor as relevance_assessor

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
        adjacent_search_steps,
        relevance_test_budget,
        community_relevance_tests,
        relevance_test_batch_size,
        irrelevant_community_restart,
        chunk_progress_callback=None,
        answer_progress_callback=None,
        chunk_callback=None,
        answer_callback=None
    ):
    answer_format = answer_schema.answer_format
    answer_object = {
        "question": question,
        "title": "",
        "claims": "",
        "content_id_sequence": [],
        "content_items": [],
        "conclusion": ""
    }
    answer_stream = []
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

    yes_id = tiktoken.get_encoding('o200k_base').encode('Yes')[0]
    no_id = tiktoken.get_encoding('o200k_base').encode('No')[0]
    logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}

    if chunk_progress_callback is not None:
        chunk_progress_callback(helper_functions.get_test_progress(test_history))
        
    aq_embedding = np.array(
        embedder.embed_store_one(
            question, embedding_cache
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

    chunk_to_communities = defaultdict(set)
    community_to_chunks = defaultdict(set)
    for chunk, concepts in chunk_to_concepts.items():
        for concept in concepts:
            if concept in concept_to_community.keys():
                community = concept_to_community[concept]
                chunk_to_communities[chunk].add(community)
                community_to_chunks[community].add(chunk)
    semantic_search_chunks = [x[1] for x in cosine_distances]
    community_sequence = []
    community_to_semantic_search_chunks = defaultdict(list)
    community_mean_rank = []
    for community, chunks in community_to_chunks.items():
        mean_rank = np.mean(sorted([semantic_search_chunks.index(c) for c in chunks])[:community_relevance_tests])
        community_mean_rank.append((community, mean_rank))
    community_sequence = [x[0] for x in sorted(community_mean_rank, key=lambda x: x[1])]

    for chunk in semantic_search_chunks:
        chunk_communities = sorted(chunk_to_communities[chunk], key=lambda x : len(community_to_concepts[x]), reverse=True)
        if len(chunk_communities) > 0:
            assigned_community = sorted(chunk_communities, key=lambda x: community_sequence.index(x))[0]
            community_to_semantic_search_chunks[assigned_community].append(chunk)

    successive_irrelevant = 0
    eliminated_communities = set()

    while len(test_history) < relevance_test_budget:
        print(f'New loop after {len(test_history)} tests')
        relevant_this_loop = False
        narrowed_community_sequence = community_sequence #[c for c in community_sequence if c not in eliminated_communities]
        for community in narrowed_community_sequence:
            relevant, seen = helper_functions.test_history_elements(test_history)
            unseen_chunks = [c for c in community_to_semantic_search_chunks[community] if c not in seen][:community_relevance_tests]
            if len(unseen_chunks) > 0:
                is_relevant = relevance_assessor.assess_relevance(
                    ai_configuration=ai_configuration,
                    search_label=f'community {community}',
                    search_chunks=unseen_chunks,
                    question=question,
                    logit_bias=logit_bias,
                    relevance_test_budget=relevance_test_budget,
                    relevance_test_batch_size=relevance_test_batch_size,
                    test_history=test_history,
                    progress_callback=chunk_progress_callback,
                    chunk_callback=chunk_callback,
                )
                relevant_this_loop |= is_relevant
                print(f'Community {community}: {is_relevant}')
                if not is_relevant:
                    eliminated_communities.add(community)
                    successive_irrelevant += 1
                    if successive_irrelevant == irrelevant_community_restart:
                        successive_irrelevant = 0
                        print(f'{successive_irrelevant} successive irrelevant communities; breaking')
                        break
                else:
                    successive_irrelevant = 0
        if not relevant_this_loop:
            print('Nothing relevant this loop')
            break

    relevant, seen = helper_functions.test_history_elements(test_history)

    # Finally, we do detail/adjacent search, which is a search of the chunks adjacent to the relevant chunks.
    adjacent_sources = list(relevant)
    adjacent_targets = set()
    for c in adjacent_sources:
        adjacent_targets.update(helper_functions.get_adjacent_chunks(c, previous_chunk, next_chunk, adjacent_search_steps))
    adjacent_search_chunks = [x for x in adjacent_targets if x not in seen]
    print(f'Adjacent: {adjacent_search_chunks}')
    relevance_assessor.assess_relevance(
        ai_configuration=ai_configuration,
        search_label='detail',
        search_chunks=adjacent_search_chunks,
        question=question,
        logit_bias=logit_bias,
        relevance_test_budget=relevance_test_budget,
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

# def answer_question_depth_breadth_detail(
#         ai_configuration,
#         question,
#         text_to_chunks,
#         chunk_to_concepts,
#         concept_to_chunks,
#         text_to_vectors,
#         concept_graph,
#         community_to_concepts,
#         concept_to_community,
#         previous_chunk,
#         next_chunk,
#         embedder,
#         embedding_cache,
#         answer_batch_size,
#         select_logit_bias,
#         semantic_search_depth,
#         adjacent_search_steps,
#         community_search_breadth,
#         relevance_test_budget,
#         relevance_test_batch_size,
#         augment_top_concepts,
#         chunk_progress_callback=None,
#         answer_progress_callback=None,
#         chunk_callback=None,
#         answer_callback=None
#     ):
#     answer_format = answer_schema.answer_format
#     answer_object = {
#         "question": question,
#         "title": "",
#         "claims": "",
#         "content_id_sequence": [],
#         "content_items": [],
#         "conclusion": ""
#     }
#     answer_stream = []
#     adjacent_processed = set()
#     sorted_chunks = []
#     test_history = []
#     answer_history = []
#     for text, chunks in text_to_chunks.items():
#         for cx, chunk in enumerate(chunks):
#             sorted_chunks.append(chunk)
#     all_units = []
    
#     for text, chunks in text_to_chunks.items():
#         for cx, chunk in enumerate(chunks):
#             all_units.append(
#                 (
#                     text,
#                     chunk,
#                     text_to_vectors[text][cx]
#                 )
#             )

#     yes_id = tiktoken.get_encoding('o200k_base').encode('Yes')[0]
#     no_id = tiktoken.get_encoding('o200k_base').encode('No')[0]
#     logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}
#     if chunk_progress_callback is not None:
#         chunk_progress_callback(helper_functions.get_test_progress(test_history))
#     last_round_matched_concepts = Counter()

#     # We repeat the query matching process for a fixed number of iterations, each time with a potentially different augmented question.
#     # This allows information gathered from prior cycles to steer the search process into different parts of the semantic space.
#     # Potentially useful for multi-hop queries.
#     while len(test_history) < relevance_test_budget:
        
#         augmented_question = f'Question: {question}'
#         # We augment the question with the top concepts matched in the previous iterations.
#         if len(last_round_matched_concepts) > 0:
#             augmented_question += f'; Matched Concepts: {"; ".join([x[0] for x in last_round_matched_concepts.most_common(augment_top_concepts)])}'
#             last_round_matched_concepts = Counter()
#         aq_embedding = np.array(
#             embedder.embed_store_one(
#                 augmented_question, embedding_cache
#             )
#         )
#         relevant, seen, terminate = helper_functions.test_history_elements(test_history)
#         cosine_distances = sorted(
#             [
#                 (t, c, scipy.spatial.distance.cosine(aq_embedding, v))
#                 for (t, c, v) in all_units if c not in seen
#             ],
#             key=lambda x: x[2],
#             reverse=False,
#         )

#         # We first do semantic search, analyze the matching chunks for relevance, and .
#         semantic_search_chunks = [x[1] for x in cosine_distances[:semantic_search_depth]]
#         relevance_assessor.assess_relevance(
#             ai_configuration=ai_configuration,
#             search_label='depth',
#             search_chunks=semantic_search_chunks,
#             question=augmented_question,
#             logit_bias=logit_bias,
#             relevance_test_budget=relevance_test_budget,
#             relevance_test_batch_size=relevance_test_batch_size,
#             test_history=test_history,
#             progress_callback=chunk_progress_callback,
#             chunk_callback=chunk_callback,
#         )
#         relevant, seen, terminate = helper_functions.test_history_elements(test_history)
#         if terminate:
#             break

#         # Next, we do breadth/community search, which is a search of the unseen chunks most representative of the communities
#         # whose concepts are contained in current chunks
#         current_concepts = set()
        
#         for chunk in relevant:
#             concepts = chunk_to_concepts[chunk]
#             current_concepts.update(concepts)

#         last_round_matched_concepts.update(current_concepts)
#         community_concept_counts = {}
#         for community in community_to_concepts.keys():
#             community_concept_counts[community] = 0
#         for concept in current_concepts:
#             if concept not in concept_to_community.keys():
#                 continue
#             community = concept_to_community[concept]
#             community_concept_counts[community] += concept_graph.nodes[concept]['count']
#         sorted_communities = sorted(
#             community_concept_counts.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         # Rank chunks within a community based on how representative they are of the community
#         # and how much they overlap with concepts relevant to the current search
#         community_to_ranked_chunks = {}
#         for community, community_concepts in community_to_concepts.items():
#             community_chunks = set()
#             for concept in community_concepts:
#                 candidate_chunks = concept_to_chunks[concept]
#                 filtered_chunks = [c for c in candidate_chunks if c not in community_chunks]
#                 community_chunks.update(filtered_chunks)
#             community_to_ranked_chunks[community] = sorted(
#                 community_chunks,
#                 key=lambda x: len(set(chunk_to_concepts[x]).intersection(community_concepts)\
#                                   .union(set(chunk_to_concepts[x]).intersection(current_concepts))),
#                 reverse=True
#             )

#         community_search_chunks = []
#         can_continue = True
#         while len(community_search_chunks) < community_search_breadth and can_continue:
#             can_continue = False
#             for community, _ in sorted_communities:
#                 unseen_community_chunks = community_to_ranked_chunks[community]
#                 unseen_community_chunks = [c for c in unseen_community_chunks if c not in seen and c not in community_search_chunks]
#                 if len(unseen_community_chunks) > 0:
#                     community_search_chunks.append(unseen_community_chunks[0])
#                 if len(community_search_chunks) > 1:
#                     can_continue = True
#                 if len(community_search_chunks) == community_search_breadth:
#                     break
            
#         relevance_assessor.assess_relevance(
#             ai_configuration=ai_configuration,
#             search_label='breadth',
#             search_chunks=community_search_chunks,
#             question=augmented_question,
#             logit_bias=logit_bias,
#             relevance_test_budget=relevance_test_budget,
#             relevance_test_batch_size=relevance_test_batch_size,
#             test_history=test_history,
#             progress_callback=chunk_progress_callback,
#             chunk_callback=chunk_callback,
#         )
#         relevant, seen, terminate = helper_functions.test_history_elements(test_history)
#         if terminate:
#             break

#         # Finally, we do detail/adjacent search, which is a search of the chunks adjacent to the relevant chunks.
#         adjacent_sources = [c for c in relevant if c not in adjacent_processed]
#         adjacent_targets = set()
#         for c in adjacent_sources:
#             adjacent_targets.update(helper_functions.get_adjacent_chunks(c, previous_chunk, next_chunk, adjacent_search_steps))
#         adjacent_search_chunks = [x for x in adjacent_targets if x not in seen]
#         adjacent_processed.update(adjacent_search_chunks)

#         relevance_assessor.assess_relevance(
#             ai_configuration=ai_configuration,
#             search_label='detail',
#             search_chunks=adjacent_search_chunks,
#             question=augmented_question,
#             logit_bias=logit_bias,
#             relevance_test_budget=relevance_test_budget,
#             relevance_test_batch_size=relevance_test_batch_size,
#             test_history=test_history,
#             progress_callback=chunk_progress_callback,
#             chunk_callback=chunk_callback,
#         )
#         relevant, seen, terminate = helper_functions.test_history_elements(test_history)
#         if terminate:
#             break

#     relevant, seen, terminate = helper_functions.test_history_elements(test_history)
#     relevant.sort(key=lambda x: sorted_chunks.index(x))

#     answer_builder.generate_answers(
#         ai_configuration=ai_configuration,
#         answer_object=answer_object,
#         answer_format=answer_format,
#         process_chunks=relevant,
#         answer_batch_size=answer_batch_size,
#         answer_stream=answer_stream,
#         answer_callback=answer_callback,
#         answer_history=answer_history,
#         progress_callback=answer_progress_callback,
#     )
#     return (
#         relevant, 
#         answer_stream,
#         helper_functions.get_test_progress(test_history),
#         helper_functions.get_answer_progress(answer_history)
#     )