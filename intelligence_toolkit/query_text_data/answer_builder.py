# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import re
from json import loads, dumps
import asyncio
import string
from tqdm.asyncio import tqdm_asyncio
from collections import defaultdict

import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.helper_functions as helper_functions
import intelligence_toolkit.query_text_data.answer_schema as answer_schema
import intelligence_toolkit.query_text_data.prompts as prompts
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
from intelligence_toolkit.query_text_data.classes import AnswerObject


def _split_on_multiple_delimiters(string, delimiters):
    # Create a regular expression pattern with the delimiters
    pattern = "|".join(map(re.escape, delimiters))
    # Split the string using the pattern
    return re.split(pattern, string)


def extract_and_link_chunk_references(text, link=True):
    source_spans = list(re.finditer(r"\[source: ([^\]]+)\]", text, re.MULTILINE))
    references = set()
    for source_span in source_spans:
        old_span = source_span.group(0)
        new_span = "[source: "
        # split on , or ; and remove whitespace
        parts = [
            x.strip()
            for x in _split_on_multiple_delimiters(source_span.group(1), [",", ";"])
        ]
        matched_parts = [x for x in parts if re.match(r"^\d+$", x)]
        references.update(matched_parts)
        if link:
            new_span += (
                ", ".join([f"[{part}](#source-{part})" for part in matched_parts]) + "]"
            )
            text = text.replace(old_span, new_span)
    references = [int(cid) for cid in references if cid.isdigit()]
    references = sorted(references)
    return text, references


def create_cid_to_label(processed_chunks):
    cid_to_label = {}
    for cid, text in enumerate(processed_chunks.cid_to_text.values()):
        chunk = loads(text)
        cid_to_label[cid] = f"{chunk['title']} ({chunk['chunk_id']})"
    return cid_to_label


async def answer_query(
    ai_configuration,
    query,
    expanded_query,
    processed_chunks,
    relevant_cids,
    cid_to_vector,
    embedder,
    embedding_cache,
    answer_config,
):
    cid_to_label = create_cid_to_label(processed_chunks)
    target_clusters = len(relevant_cids) // answer_config.target_chunks_per_cluster
    if len(relevant_cids) / answer_config.target_chunks_per_cluster > target_clusters:
        target_clusters += 1
    clustered_cids = cluster_cids(relevant_cids, cid_to_vector, target_clusters)
    clustered_texts = [
        [f"{cid}: {processed_chunks.cid_to_text[cid]}" for cid in cids]
        for cids in clustered_cids.values()
    ]
    source_to_supported_claims = defaultdict(set)
    source_to_contradicted_claims = defaultdict(set)
    net_new_sources = 0
    if answer_config.extract_claims:
        batched_extraction_messages = [
            utils.prepare_messages(
                prompts.claim_extraction_prompt,
                {"chunks": texts, "query": expanded_query},
            )
            for texts in clustered_texts
        ]

        extracted_claims = await utils.map_generate_text(
            ai_configuration,
            batched_extraction_messages,
            response_format=answer_schema.claim_extraction_format,
        )
        json_extracted_claims = []
        for claim in extracted_claims:
            try:
                json_claim = loads(claim)
                json_extracted_claims.append(json_claim)
            except Exception as e:
                print(f"Error loading claim as JSON: {claim}")
        tasks = []
        claim_context_to_claim_supporting_sources = defaultdict(
            lambda: defaultdict(set)
        )
        claim_context_to_claim_contradicting_sources = defaultdict(
            lambda: defaultdict(set)
        )

        claims_to_embed = {}

        for claim_sets in json_extracted_claims:
            for claim_set in claim_sets["claim_analysis"]:
                claim_context = claim_set["claim_context"]
                for claim in claim_set["claims"]:
                    claim_statement = claim["claim_statement"]
                    claims_to_embed[len(claims_to_embed)] = (
                        f"{claim_statement} (context: {claim_context})"
                    )
                    for source in claim["supporting_sources"]:
                        source_to_supported_claims[source].add(
                            (claim_context, claim_statement)
                        )
                    for source in claim["contradicting_sources"]:
                        source_to_contradicted_claims[source].add(
                            (claim_context, claim_statement)
                        )
        cix_to_vector = await helper_functions.embed_queries(
            claims_to_embed, embedder, cache_data=embedding_cache
        )

        claim_to_vector = {
            claims_to_embed[cix]: vector for cix, vector in cix_to_vector.items()
        }
        units = sorted(
            [(cid, vector) for cid, vector in (cid_to_vector.items())],
            key=lambda x: x[0],
        )
        neighbours = NearestNeighbors(
            n_neighbors=answer_config.claim_search_depth, metric="cosine"
        ).fit([vector for cid, vector in units])

        for claim_sets in json_extracted_claims:
            for claim_set in claim_sets["claim_analysis"]:
                claim_context = claim_set["claim_context"]
                for claim in claim_set["claims"]:
                    claim_statement = claim["claim_statement"]
                    claim_key = f"{claim_statement} (context: {claim_context})"
                    claim_context_to_claim_supporting_sources[claim_context][
                        claim_statement
                    ].update(claim["supporting_sources"])
                    claim_context_to_claim_contradicting_sources[claim_context][
                        claim_statement
                    ].update(claim["contradicting_sources"])
                    if claim_key in claim_to_vector:
                        tasks.append(
                            asyncio.create_task(
                                requery_claim(
                                    ai_configuration,
                                    expanded_query,
                                    units,
                                    neighbours,
                                    claim_to_vector[claim_key],
                                    claim_context,
                                    claim_statement,
                                    processed_chunks.cid_to_text,
                                    cid_to_label,
                                )
                            )
                        )
                    else:
                        print(f"No vector for claim: {claim_key}")
        print(f"Running {len(tasks)} requery tasks")
        results_list = await tqdm_asyncio.gather(*tasks)
        for (
            claim_context,
            claim_statement,
            supporting_sources,
            contradicting_sources,
        ) in results_list:
            claim_context_to_claim_supporting_sources[claim_context][
                claim_statement
            ].update(supporting_sources)
            claim_context_to_claim_contradicting_sources[claim_context][
                claim_statement
            ].update(contradicting_sources)
        for (
            claim_context,
            claims_to_support,
        ) in claim_context_to_claim_supporting_sources.items():
            for claim_statement, supporting_sources in claims_to_support.items():
                for source in supporting_sources:
                    source_to_supported_claims[source].add(
                        (claim_context, claim_statement)
                    )
        for (
            claim_context,
            claims_to_contradict,
        ) in claim_context_to_claim_contradicting_sources.items():
            for claim_statement, contradicting_sources in claims_to_contradict.items():
                for source in contradicting_sources:
                    source_to_contradicted_claims[source].add(
                        (claim_context, claim_statement)
                    )

        all_sources = set(source_to_supported_claims.keys()).union(
            set(source_to_contradicted_claims.keys())
        )
        net_new_sources = len(all_sources) - len(relevant_cids)

        all_claims = set()
        for (
            claim_context,
            claims_to_support,
        ) in claim_context_to_claim_supporting_sources.items():
            for claim_statement in claims_to_support.keys():
                all_claims.add((claim_context, claim_statement))
        for (
            claim_context,
            claims_to_contradict,
        ) in claim_context_to_claim_contradicting_sources.items():
            for claim_statement in claims_to_contradict.keys():
                all_claims.add((claim_context, claim_statement))
        print(f"Used {len(all_claims)} claims")

        claim_summaries = []

        for (
            claim_context,
            claims_to_support,
        ) in claim_context_to_claim_supporting_sources.items():
            for claim_statement, supporting_sources in claims_to_support.items():
                contradicting_sources = claim_context_to_claim_contradicting_sources[
                    claim_context
                ][claim_statement]
                claim_summaries.append(
                    {
                        "claim_context": claim_context,
                        "claims": [
                            {
                                "claim_statement": claim_statement,
                                "claim_attribution": "",
                                "supporting_sources": sorted(supporting_sources),
                                "contradicting_sources": sorted(contradicting_sources),
                            }
                        ],
                        "sources": {
                            source: text
                            for source, text in processed_chunks.cid_to_text.items()
                            if source in supporting_sources.union(contradicting_sources)
                        },
                    }
                )

        def extract_relevant_chunks(claims):
            relevant_cids = set()
            for claim in claims["claims"]:
                ss = set(claim["supporting_sources"])
                cs = set(claim["contradicting_sources"])
                relevant_cids.update(ss.union(cs))
            relevant_cids = sorted(relevant_cids)
            return [
                f"{cid}: {processed_chunks.cid_to_text[cid]}" for cid in relevant_cids
            ]

        batched_summarization_messages = [
            utils.prepare_messages(
                prompts.claim_summarization_prompt,
                {
                    "analysis": dumps(claims, ensure_ascii=False, indent=2),
                    "chunks": extract_relevant_chunks(claims),
                    "query": expanded_query,
                },
            )
            for i, claims in enumerate(claim_summaries)
        ]
    else:
        batched_summarization_messages = [
            utils.prepare_messages(
                prompts.claim_summarization_prompt,
                {"analysis": [], "chunks": clustered_texts[i], "query": expanded_query},
            )
            for i in range(len(clustered_texts))
        ]

    summarized_claims = await utils.map_generate_text(
        ai_configuration,
        batched_summarization_messages,
        response_format=answer_schema.claim_summarization_format,
    )
    content_items_list = []
    for content_items in summarized_claims:
        json_content_items = loads(content_items)
        for item in json_content_items["content_items"]:
            content_items_list.append(item)
    content_items_dict = {i: v for i, v in enumerate(content_items_list)}
    content_items_context = dumps(content_items_dict, indent=2, ensure_ascii=False)
    content_item_messages = utils.prepare_messages(
        prompts.content_integration_prompt,
        {"content": content_items_context, "query": query},
    )
    content_structure = loads(
        utils.generate_text(
            ai_configuration,
            content_item_messages,
            response_format=answer_schema.content_integration_format,
        )
    )

    report, references, matched_chunks = build_report_markdown(
        query,
        expanded_query,
        content_items_dict,
        content_structure,
        processed_chunks.cid_to_text,
        source_to_supported_claims,
        source_to_contradicted_claims,
    )
    return AnswerObject(
        extended_answer=report,
        references=references,
        referenced_chunks=matched_chunks,
        net_new_sources=net_new_sources,
    )


def build_report_markdown(
    query,
    expanded_query,
    content_items_dict,
    content_structure,
    cid_to_text,
    source_to_supported_claims,
    source_to_contradicted_claims,
):
    text_jsons = [loads(text) for text in cid_to_text.values()]
    matched_chunks = {
        f"{text['title']} ({text['chunk_id']})": text for text in text_jsons
    }
    home_link = "#report"
    report = f'# Report\n\n## Query\n\n*{query}*\n\n## Expanded Query\n\n*{expanded_query}*\n\n## Answer\n\n{content_structure["answer"]}\n\n## Analysis\n\n### {content_structure["report_title"]}\n\n{content_structure["report_summary"]}\n\n'
    for theme in content_structure["theme_order"]:
        report += f'#### Theme: {theme["theme_title"]}\n\n{theme["theme_summary"]}\n\n'
        for item_id in theme["content_id_order"]:
            item = content_items_dict[item_id]
            report += f'##### {item["content_title"]}\n\n{item["content_summary"]}\n\n{item["content_commentary"]}\n\n'
        report += f'##### AI theme commentary\n\n{theme["theme_commentary"]}\n\n'
    report += (
        f'#### AI report commentary\n\n{content_structure["report_commentary"]}\n\n'
    )
    report, references = extract_and_link_chunk_references(report)
    print(f"Extracted references: {references}")
    report += f"## Sources\n\n"
    for cid in references:
        if cid in cid_to_text.keys():
            supports_claims = source_to_supported_claims[cid]
            contradicts_claims = source_to_contradicted_claims[cid]
            supports_claims_str = "- " + "\n- ".join(
                [claim_statement for _, claim_statement in supports_claims]
            )
            contradicts_claims_str = "- " + "\n- ".join(
                [claim_statement for _, claim_statement in contradicts_claims]
            )
            chunk = loads(cid_to_text[cid])
            report += f'#### Source {cid}\n\n<details>\n\n##### Text chunk: {chunk["title"]} ({chunk["chunk_id"]})\n\n{chunk["text_chunk"]}\n\n'
            if len(supports_claims) > 0:
                report += f"##### Supports claims\n\n{supports_claims_str}\n\n"
            if len(contradicts_claims) > 0:
                report += f"##### Contradicts claims\n\n{contradicts_claims_str}\n\n"
            report += f"</details>\n\n[Back to top]({home_link})\n\n"
        else:
            print(f"No match for {cid}")

    return report, references, matched_chunks


def cluster_cids(relevant_cids, cid_to_vector, target_clusters):
    clustered_cids = {}
    if len(relevant_cids) > 0:
        # use k-means clustering to group relevant cids into target_clusters clusters
        cids = []
        vectors = []
        for relevant_cid in relevant_cids:
            if relevant_cid in cid_to_vector:
                cids.append(relevant_cid)
                vectors.append(cid_to_vector[relevant_cid])
        kmeans = cluster.KMeans(n_clusters=target_clusters)
        kmeans.fit(vectors)
        cluster_assignments = kmeans.predict(vectors)

        for i, cid in enumerate(cids):
            cluster_assignment = cluster_assignments[i]
            if cluster_assignment not in clustered_cids:
                clustered_cids[cluster_assignment] = []
            clustered_cids[cluster_assignment].append(cid)
    return clustered_cids


async def requery_claim(
    ai_configuration,
    query,
    units,
    neighbours,
    claim_embedding,
    claim_context,
    claim_statement,
    cid_to_text,
    cid_to_label,
):
    contextualized_claim = f"{claim_statement} (context: {claim_context})"
    # Find the nearest neighbors of the claim embedding
    indices = neighbours.kneighbors([claim_embedding], return_distance=False)
    cids = [units[i][0] for i in indices[0]]
    # cosine_distances = sorted(
    #     [
    #         (cid, scipy.spatial.distance.cosine(claim_embedding, vector))
    #         for (cid, vector) in all_units
    #     ],
    #     key=lambda x: x[1],
    #     reverse=False,
    # )
    # batch cids into batches of size batch_size
    # cids = [cid for cid, dist in cosine_distances[:search_depth]]
    chunks = dumps(
        {cid: cid_to_text[cid] for i, cid in enumerate(cids)},
        ensure_ascii=False,
        indent=2,
    )
    messages = utils.prepare_messages(
        prompts.claim_requery_prompt,
        {"query": query, "claim": contextualized_claim, "chunks": chunks},
    )
    response = await utils.generate_text_async(
        ai_configuration, messages, response_format=answer_schema.claim_requery_format
    )

    response_json = loads(response)
    supporting_sources = response_json["supporting_sources"]
    contradicting_sources = response_json["contradicting_sources"]
    print(
        f"Claim: {contextualized_claim} has supporting sources: {supporting_sources} and contradicting sources: {contradicting_sources}"
    )
    return claim_context, claim_statement, supporting_sources, contradicting_sources