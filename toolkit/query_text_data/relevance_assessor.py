# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import asyncio
from json import loads
from collections import defaultdict
import numpy as np
import scipy.spatial.distance
import tiktoken
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
    num_adjacent,
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
        if len(test_history) + len(mapped_messages) + num_adjacent > relevance_test_budget:
            mapped_messages = mapped_messages[:relevance_test_budget - len(test_history)]
        mapped_responses = await utils.map_generate_text(
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

async def detect_relevant_chunks(
    ai_configuration,
    question,
    cid_to_text,
    cid_to_concepts,
    cid_to_vector,
    community_to_concepts,
    concept_to_community,
    previous_cid,
    next_cid,
    embedder,
    embedding_cache,
    select_logit_bias,
    adjacent_search_steps,
    relevance_test_budget,
    community_relevance_tests,
    relevance_test_batch_size,
    irrelevant_community_restart,
    chunk_progress_callback=None,
    chunk_callback=None,
):
    test_history = []

    all_units = []
    all_units = sorted([(cid, vector) for cid, vector in (cid_to_vector.items())], key=lambda x: x[0])


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
    relevant, seen, adjacent = helper_functions.test_history_elements(test_history, previous_cid, next_cid, adjacent_search_steps)

    cosine_distances = sorted(
        [
            (cid, scipy.spatial.distance.cosine(aq_embedding, vector))
            for (cid, vector) in all_units if cid not in seen
        ],
        key=lambda x: x[1],
        reverse=False,
    )

    cid_to_communities = defaultdict(set)
    community_to_cids = defaultdict(set)
    for cid, concepts in cid_to_concepts.items():
        for concept in concepts:
            if concept in concept_to_community.keys():
                community = concept_to_community[concept]
                cid_to_communities[cid].add(community)
                community_to_cids[community].add(cid)
    semantic_search_cids = [x[0] for x in cosine_distances]
    community_sequence = []
    community_to_semantic_search_cids = defaultdict(list)
    community_mean_rank = []

    for community, cids in community_to_cids.items():
        mean_rank = np.mean(sorted([semantic_search_cids.index(c) for c in cids])[:community_relevance_tests])
        community_mean_rank.append((community, mean_rank))
    community_sequence = [x[0] for x in sorted(community_mean_rank, key=lambda x: x[1])]

    for cid in semantic_search_cids:
        chunk_communities = sorted(cid_to_communities[cid], key=lambda x : len(community_to_concepts[x]), reverse=True)
        if len(chunk_communities) > 0:
            assigned_community = sorted(chunk_communities, key=lambda x: community_sequence.index(x))[0]
            community_to_semantic_search_cids[assigned_community].append(cid)

    successive_irrelevant = 0
    eliminated_communities = set()

    while len(test_history) + len(adjacent) < relevance_test_budget:
        print(f'New loop after {len(test_history)} tests')
        relevant_this_loop = False

        for community in community_sequence:
            relevant, seen, adjacent = helper_functions.test_history_elements(test_history, previous_cid, next_cid, adjacent_search_steps)
            unseen_cids = [c for c in community_to_semantic_search_cids[community] if c not in seen][:community_relevance_tests]
            if len(unseen_cids) > 0:
                is_relevant = await assess_relevance(
                    ai_configuration=ai_configuration,
                    search_label=f"topic {community}",
                    search_cids=unseen_cids,
                    cid_to_text=cid_to_text,
                    question=question,
                    logit_bias=logit_bias,
                    relevance_test_budget=relevance_test_budget,
                    num_adjacent=len(adjacent),
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

    relevant, seen, adjacent = helper_functions.test_history_elements(test_history, previous_cid, next_cid, adjacent_search_steps)

    await assess_relevance(
        ai_configuration=ai_configuration,
        search_label="neighbours",
        search_cids=adjacent,
        cid_to_text=cid_to_text,
        question=question,
        logit_bias=logit_bias,
        relevance_test_budget=relevance_test_budget,
        num_adjacent=len(adjacent),
        relevance_test_batch_size=relevance_test_batch_size,
        test_history=test_history,
        progress_callback=chunk_progress_callback,
        chunk_callback=chunk_callback,
    )
    relevant, seen, adjacent = helper_functions.test_history_elements(test_history, previous_cid, next_cid, adjacent_search_steps)
    relevant.sort()

    return relevant, helper_functions.get_test_progress(test_history)
