# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import json

from intelligence_toolkit.AI.base_embedder import BaseEmbedder
from intelligence_toolkit.AI.classes import VectorData
from intelligence_toolkit.AI.utils import hash_text


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
    current_search = ""
    current_relevant = 0
    current_tested = 0
    total_relevant = 0
    total_tested = 0
    rounds = []
    for ix, (search, chunk, response) in enumerate(test_history):
        if search != current_search:
            if current_search != "":
                if current_relevant > 0:
                    rounds.append(
                        f"{current_search}: {current_relevant}/{current_tested}"
                    )
                else:
                    rounds.append(
                        f"<span style='color: red'>{current_search}: {current_relevant}/{current_tested}</span>"
                    )
            current_search = search
            current_relevant = 0
            current_tested = 0
        current_tested += 1
        total_tested += 1
        if response == "Yes":
            current_relevant += 1
            total_relevant += 1
    if current_search != "":
        if current_relevant > 0:
            rounds.append(f"{current_search}: {current_relevant}/{current_tested}")
        else:
            rounds.append(
                f"<span style='color: red'>{current_search}: {current_relevant}/{current_tested}</span>"
            )
    response = f"**Relevant chunks / tested chunks: {total_relevant}/{total_tested}**"
    if len(rounds) > 0:
        response += " (" + "; ".join(rounds) + ")"
    return response


def parse_history_elements(test_history, previous_cid, next_cid, adjacent_search_steps):
    relevant_list = [x[1] for x in test_history if x[2] == "Yes"]
    seen_list = [x[1] for x in test_history]
    adjacent_targets = set()
    for c in relevant_list:
        adjacent_targets.update(
            get_adjacent_chunks(c, previous_cid, next_cid, adjacent_search_steps)
        )
    adjacent_list = [x for x in adjacent_targets if x not in seen_list]
    return relevant_list, seen_list, adjacent_list


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


async def embed_queries(
    qid_to_text, text_embedder: BaseEmbedder, cache_data=True, callbacks=[]
) -> dict:
    qid_to_vector = {}
    data: list[VectorData] = []

    for qid, text in qid_to_text.items():
        data.append(
            {"hash": hash_text(text), "text": text, "additional_details": {"qid": qid}}
        )

    embedded_data = await text_embedder.embed_store_many(data, callbacks, cache_data)
    for item in embedded_data:
        # find item in data
        data_item = next((x for x in data if x["hash"] == item["hash"]), None)

        if data_item is None:
            print(f"No matching data item for {item}")
            continue

        details = json.loads(item["additional_details"])
        additional_details = data_item["additional_details"]

        if isinstance(additional_details, str):
            additional_details = json.loads(additional_details)

        qid = additional_details.get("qid")
        if qid is None:
            print(f"No qid found in additional details for {item}")
            continue

        if details.get("qid") != qid:
            details = {"qid": qid}

        qid_to_vector[qid] = item["vector"]
    return qid_to_vector