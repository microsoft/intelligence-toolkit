# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import asyncio
from json import loads
from collections import defaultdict
import numpy as np
import scipy.spatial.distance
import tiktoken
import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.helper_functions as helper_functions
import intelligence_toolkit.query_text_data.prompts as prompts
from intelligence_toolkit.query_text_data.commentary import Commentary

async def assess_relevance(
    ai_configuration,
    search_label,
    search_cids,
    cid_to_text,
    query,
    logit_bias,
    relevance_test_budget,
    num_adjacent,
    relevance_test_batch_size,
    test_history,
    progress_callback,
    chunk_callback,
    commentary
):
    batched_cids = [
        search_cids[i : i + relevance_test_batch_size]
        for i in range(0, len(search_cids), relevance_test_batch_size)
    ]
    batched_texts = [[cid_to_text[cid] for cid in batch] for batch in batched_cids]
    batched_messages = [
        [
            utils.prepare_messages(
                prompts.chunk_relevance_prompt, {"chunk": chunk, "query": query}
            )
            for chunk in batch
        ]
        for batch in batched_texts
    ]
    is_relevant = False
    for mx, mapped_messages in enumerate(batched_messages):
        cid_batch = batched_cids[mx]
        if (
            len(test_history) + len(mapped_messages) + num_adjacent
            > relevance_test_budget
        ):
            mapped_messages = mapped_messages[
                : relevance_test_budget - len(test_history)
            ]
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
            chunk_callback,
            commentary,
        )
        is_relevant = num_relevant > 0
        if not is_relevant:  # No relevant chunks found in this batch; terminate early
            break
    return is_relevant


def process_relevance_responses(
    search_label,
    search_cids,
    cid_to_text,
    mapped_responses,
    test_history,
    progress_callback,
    chunk_callback,
    commentary,
):
    tested_relevant = []
    for r, c in zip(mapped_responses, search_cids):
        if c not in [x[1] for x in test_history]:
            test_history.append((search_label, c, r))
            if r == "Yes":
                tested_relevant.append(c)
    if progress_callback is not None:
        progress_callback(helper_functions.get_test_progress(test_history))
    relevant_list = [x[1] for x in test_history if x[2] == "Yes"]
    if chunk_callback is not None:
        chunk_callback([cid_to_text[cid] for cid in relevant_list])
    
    if commentary is not None and len(tested_relevant) > 0:
        relevant_texts = {cid: cid_to_text[cid] for cid in tested_relevant}
        commentary.update_analysis(relevant_texts)
    return len(tested_relevant)


async def detect_relevant_chunks(
    ai_configuration,
    query,
    processed_chunks,
    cid_to_vector,
    embedder,
    embedding_cache,
    chunk_search_config,
    chunk_progress_callback=None,
    chunk_callback=None,
    analysis_callback=None,
    commentary_callback=None,
):
    commentary = Commentary(ai_configuration, query, analysis_callback, commentary_callback) if analysis_callback is not None and commentary_callback is not None else None
    test_history = []
    all_units = sorted(
        [(cid, vector) for cid, vector in (cid_to_vector.items())], key=lambda x: x[0]
    )

    yes_id = tiktoken.get_encoding("o200k_base").encode("Yes")[0]
    no_id = tiktoken.get_encoding("o200k_base").encode("No")[0]
    select_logit_bias = 5
    logit_bias = {yes_id: select_logit_bias, no_id: select_logit_bias}

    if chunk_progress_callback is not None:
        chunk_progress_callback(helper_functions.get_test_progress(test_history))

    aq_embedding = np.array(embedder.embed_store_one(query, embedding_cache))
    relevant, seen, adjacent = helper_functions.test_history_elements(
        test_history,
        processed_chunks.previous_cid,
        processed_chunks.next_cid,
        chunk_search_config.adjacent_test_steps,
    )
    cosine_distances = sorted(
        [
            (cid, scipy.spatial.distance.cosine(aq_embedding, vector))
            for (cid, vector) in all_units
            if cid not in seen
        ],
        key=lambda x: x[1],
        reverse=False,
    )
    semantic_search_cids = [x[0] for x in cosine_distances]
    # print(f"Top semantic search cids: {semantic_search_cids[:100]}")
    level_to_community_sequence = {}
    max_level = max([hc.level for hc in processed_chunks.hierarchical_communities])
    concept_to_level_to_community = defaultdict(dict)
    level_to_community_to_candidate_cids = defaultdict(lambda: defaultdict(set))
    level_to_community_to_cids = defaultdict(lambda: defaultdict(list))
    level_to_cid_to_communities = defaultdict(lambda: defaultdict(set))
    community_to_parent = {}
    for hc in processed_chunks.hierarchical_communities:
        concept_to_level_to_community[hc.node][hc.level] = (
            processed_chunks.community_to_label[hc.cluster]
        )
        if hc.parent_cluster is not None:
            community_to_parent[processed_chunks.community_to_label[hc.cluster]] = (
                processed_chunks.community_to_label[hc.parent_cluster]
            )
    cid_to_level_to_communities = defaultdict(lambda: defaultdict(set))
    for level in range(0, max_level + 1):
        for cid, concepts in processed_chunks.cid_to_concepts.items():
            for concept in concepts:
                if concept in concept_to_level_to_community.keys():
                    if level in concept_to_level_to_community[concept].keys():
                        community = concept_to_level_to_community[concept][level]
                        cid_to_level_to_communities[cid][level].add(community)
                        level_to_cid_to_communities[level][cid].add(community)
                        level_to_community_to_candidate_cids[level][community].add(cid)
                    else:
                        # use the community from the previous level
                        if level - 1 in concept_to_level_to_community[concept].keys():
                            community = concept_to_level_to_community[concept][
                                level - 1
                            ]
                            cid_to_level_to_communities[cid][level].add(community)
                            level_to_cid_to_communities[level][cid].add(community)
                            level_to_community_to_candidate_cids[level][community].add(
                                cid
                            )

        community_sequence = []
        community_mean_rank = []

        for community, cids in level_to_community_to_candidate_cids[level].items():
            filtered_cids = [c for c in cids if c in semantic_search_cids]
            mean_rank = np.mean(
                sorted([semantic_search_cids.index(c) for c in filtered_cids])[
                    : chunk_search_config.community_ranking_chunks
                ]
            )
            community_mean_rank.append((community, mean_rank))
        community_sequence = [
            x[0] for x in sorted(community_mean_rank, key=lambda x: x[1])
        ]
        # print(f"Level {level} community sequence: {community_sequence}")
        level_to_community_sequence[level] = community_sequence

        for cid in semantic_search_cids:
            chunk_communities = cid_to_level_to_communities[cid][level]
            if len(chunk_communities) > 0:
                assigned_community = sorted(
                    chunk_communities, key=lambda x: community_sequence.index(x)
                )[0]
                if cid not in level_to_community_to_cids[level][assigned_community]:
                    level_to_community_to_cids[level][assigned_community].append(cid)

    for level, community_to_cids in level_to_community_to_cids.items():
        for community, cids in community_to_cids.items():
            cids.sort(key=lambda x: semantic_search_cids.index(x))

    # Set level -1 as everything in the dataset
    level_to_community_sequence[-1] = ["1"]
    level_to_community_to_cids[-1]["1"] = semantic_search_cids
    for concept, level_to_community in concept_to_level_to_community.items():
        level_to_community[-1] = "1"

    for cid, level_to_community in cid_to_level_to_communities.items():
        level_to_community[-1] = "1"

    successive_irrelevant = 0
    eliminated_communities = set()
    current_level = -1

    while len(test_history) + len(adjacent) < chunk_search_config.relevance_test_budget:
        # print(f"New level {current_level} loop after {len(test_history)} tests")
        relevant_this_loop = False

        community_sequence = []
        for community in level_to_community_sequence[current_level]:
            if community in community_to_parent.keys():
                parent = community_to_parent[community]
                if parent not in eliminated_communities:
                    community_sequence.append(community)
                else:
                    eliminated_communities.add(community)
                    # print(f"Eliminated community {community} due to parent {parent}")
            else:
                community_sequence.append(community)
        # print(f"Community sequence: {community_sequence}")
        community_to_cids = level_to_community_to_cids[current_level]
        for community in community_sequence:
            relevant, seen, adjacent = helper_functions.test_history_elements(
                test_history,
                processed_chunks.previous_cid,
                processed_chunks.next_cid,
                chunk_search_config.adjacent_test_steps,
            )
            unseen_cids = [c for c in community_to_cids[community] if c not in seen][
                : chunk_search_config.community_relevance_tests
            ]
            if len(unseen_cids) > 0:
                # print(
                #     f"Assessing relevance for community {community} with chunks {unseen_cids}"
                # )
                is_relevant = await assess_relevance(
                    ai_configuration=ai_configuration,
                    search_label=f"topic {community}",
                    search_cids=unseen_cids,
                    cid_to_text=processed_chunks.cid_to_text,
                    query=query,
                    logit_bias=logit_bias,
                    relevance_test_budget=chunk_search_config.relevance_test_budget,
                    num_adjacent=len(adjacent),
                    relevance_test_batch_size=chunk_search_config.relevance_test_batch_size,
                    test_history=test_history,
                    progress_callback=chunk_progress_callback,
                    chunk_callback=chunk_callback,
                    commentary=commentary
                )
                if len(test_history) + len(adjacent) >= chunk_search_config.relevance_test_budget:
                    break
                relevant_this_loop |= is_relevant
                # print(f"Community {community} relevant? {is_relevant}")
                if (
                    current_level > -1 and not is_relevant
                ):  # don't stop after failure at the root level
                    eliminated_communities.add(community)
                    successive_irrelevant += 1
                    if (
                        successive_irrelevant
                        == chunk_search_config.irrelevant_community_restart
                    ):
                        # print(
                        #     f"{successive_irrelevant} successive irrelevant communities; restarting"
                        # )
                        successive_irrelevant = 0
                        break
                else:
                    successive_irrelevant = 0
        if (
            current_level > -1 and not relevant_this_loop
        ):  # don't stop after failure at the root level
            # print("Nothing relevant this loop")
            break
        if current_level + 1 in level_to_community_sequence.keys():
            # print("Incrementing level")
            current_level += 1
        else:
            # print("Reached final level")
            pass

    relevant, seen, adjacent = helper_functions.test_history_elements(
        test_history,
        processed_chunks.previous_cid,
        processed_chunks.next_cid,
        chunk_search_config.adjacent_test_steps,
    )

    await assess_relevance(
        ai_configuration=ai_configuration,
        search_label="neighbours",
        search_cids=adjacent,
        cid_to_text=processed_chunks.cid_to_text,
        query=query,
        logit_bias=logit_bias,
        relevance_test_budget=chunk_search_config.relevance_test_budget,
        num_adjacent=len(adjacent),
        relevance_test_batch_size=chunk_search_config.relevance_test_batch_size,
        test_history=test_history,
        progress_callback=chunk_progress_callback,
        chunk_callback=chunk_callback,
        commentary=commentary
    )
    relevant, seen, adjacent = helper_functions.test_history_elements(
        test_history,
        processed_chunks.previous_cid,
        processed_chunks.next_cid,
        chunk_search_config.adjacent_test_steps,
    )
    relevant.sort()

    return relevant, helper_functions.get_test_progress(test_history), commentary
