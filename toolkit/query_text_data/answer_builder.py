# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import re
from json import loads, dumps
import numpy as np
import asyncio
import string
from tqdm.asyncio import tqdm_asyncio
from collections import defaultdict

import toolkit.AI.utils as utils
import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.answer_schema as answer_schema
import toolkit.query_text_data.prompts as prompts
import sklearn.cluster as cluster
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
from toolkit.query_text_data.classes import AnswerObject

def extract_chunk_references(text):
    source_spans = list(re.finditer(r'\[source: ([^\]]+)\]', text, re.MULTILINE))
    references = set()
    for source_span in source_spans:
        parts = source_span.group(1).split(', ')
        references.update(parts)
    ref_list = sorted(references)
    return ref_list

def link_chunk_references(text, references):
    for ix, reference in enumerate(references):
        text = text.replace(reference, f"[{ix+1}](#source-{ix+1})")
    return text

async def answer_question(
    ai_configuration,
    question,
    processed_chunks,
    relevant_cids,
    cid_to_vector,
    embedder,
    embedding_cache,
    answer_config,
):
    target_clusters = len(relevant_cids) // answer_config.target_chunks_per_cluster
    if len(relevant_cids) / answer_config.target_chunks_per_cluster > target_clusters:
        target_clusters += 1
    clustered_cids = cluster_cids(
        relevant_cids,
        cid_to_vector,
        target_clusters
    )
    clustered_texts = [[processed_chunks.cid_to_text[cid] for cid in cids] for cids in clustered_cids.values()]
    source_to_text = {f"{text['title']} ({text['chunk_id']})": text for text in [loads(text) for text in processed_chunks.cid_to_text.values()]}
    source_to_supported_claims = defaultdict(set)
    source_to_contradicted_claims = defaultdict(set)
    net_new_sources = 0
    if answer_config.extract_claims:
        batched_extraction_messages = [utils.prepare_messages(prompts.claim_extraction_prompt, {'chunks': texts, 'question': question}) 
                            for texts in clustered_texts]

        extracted_claims = await utils.map_generate_text(
            ai_configuration, batched_extraction_messages, response_format=answer_schema.claim_extraction_format
        )
        json_extracted_claims = []
        for claim in extracted_claims:
            try:
                json_claim = loads(claim)
                json_extracted_claims.append(json_claim)
            except Exception as e:
                print(f'Error loading claim as JSON: {claim}')
        tasks = []
        claim_context_to_claim_supporting_sources = defaultdict(lambda: defaultdict(set))
        claim_context_to_claim_contradicting_sources = defaultdict(lambda: defaultdict(set))

        claims_to_embed = {}

        for claim_sets in json_extracted_claims:
            for claim_set in claim_sets['claim_analysis']:
                claim_context = claim_set['claim_context']
                for claim in claim_set['claims']:
                    claim_statement = claim['claim_statement']
                    claims_to_embed[len(claims_to_embed)] = f"{claim_statement} (context: {claim_context})"

        cix_to_vector = await helper_functions.embed_queries(
            claims_to_embed,
            embedder,
            cache_data=embedding_cache
        )

        claim_to_vector = {claims_to_embed[cix]: vector for cix, vector in cix_to_vector.items()}
        units = sorted([(cid, vector) for cid, vector in (cid_to_vector.items())], key=lambda x: x[0])
        neighbours = NearestNeighbors(n_neighbors=answer_config.claim_search_depth, metric='cosine').fit([vector for cid, vector in units])
        cix = 0
        for claim_sets in json_extracted_claims:
            for claim_set in claim_sets['claim_analysis']:
                claim_context = claim_set['claim_context']
                for claim in claim_set['claims']:
                    claim_statement = claim['claim_statement']
                    claim_key = f"{claim_statement} (context: {claim_context})"
                    supporting_sources = set()
                    for ss in claim['supporting_sources']:
                        tt = ss['text_title']
                        for sc in ss['chunk_ids']:
                            supporting_sources.add(f"{tt} ({sc})")
                    contradicting_sources = set()
                    for cs in claim['contradicting_sources']:
                        tt = cs['text_title']
                        for sc in cs['chunk_ids']:
                            contradicting_sources.add(f"{tt} ({sc})")
                    claim_context_to_claim_supporting_sources[claim_context][claim_statement] = supporting_sources
                    claim_context_to_claim_contradicting_sources[claim_context][claim_statement] = contradicting_sources
                    if claim_key in claim_to_vector:
                        tasks.append(asyncio.create_task(requery_claim(
                            ai_configuration,
                            question,
                            units,
                            neighbours,
                            claim_to_vector[claim_key],
                            claim_context,
                            claim_statement,
                            processed_chunks.cid_to_text
                        )))
                    else:
                        print(f'No vector for claim: {claim_key}')
        print(f'Running {len(tasks)} requery tasks')
        results_list = await tqdm_asyncio.gather(*tasks)
        for claim_context, claim_statement, supporting_sources, contradicting_sources in results_list:
            claim_context_to_claim_supporting_sources[claim_context][claim_statement].update(supporting_sources)
            claim_context_to_claim_contradicting_sources[claim_context][claim_statement].update(contradicting_sources)
        
        for claim_context, claims_to_support in claim_context_to_claim_supporting_sources.items():
            for claim_statement, supporting_sources in claims_to_support.items():
                for source in supporting_sources:
                    source_to_supported_claims[source].add((claim_context, claim_statement))
        
        for claim_context, claims_to_contradict in claim_context_to_claim_contradicting_sources.items():
            for claim_statement, contradicting_sources in claims_to_contradict.items():
                for source in contradicting_sources:
                    source_to_contradicted_claims[source].add((claim_context, claim_statement))

        all_sources = set(source_to_supported_claims.keys()).union(set(source_to_contradicted_claims.keys()))
        net_new_sources = len(all_sources) - len(relevant_cids)

        claim_summaries = []
        
        for claim_context, claims_to_support in claim_context_to_claim_supporting_sources.items():
            for claim_statement, supporting_sources in claims_to_support.items():
                contradicting_sources = claim_context_to_claim_contradicting_sources[claim_context][claim_statement]
                claim_summaries.append(
                    {
                        'claim_context': claim_context,
                        'claims': [
                            {
                                'claim_statement': claim_statement,
                                'claim_attribution': '',
                                'supporting_sources': sorted(supporting_sources),
                                'contradicting_sources': sorted(contradicting_sources)
                            }
                        ],
                        'sources': {
                            source: text for source, text in source_to_text.items() if source in supporting_sources.union(contradicting_sources)
                        }
                    }
                )
        batched_summarization_messages = [utils.prepare_messages(prompts.claim_summarization_prompt, {'analysis': dumps(claims, ensure_ascii=False, indent=2), 'data': '', 'question': question}) 
                            for i, claims in enumerate(claim_summaries)]
    else:
        batched_summarization_messages = [utils.prepare_messages(prompts.claim_summarization_prompt, {'analysis': [], 'data': clustered_texts[i], 'question': question}) 
                            for i in range(len(clustered_texts))]
        
    summarized_claims = await utils.map_generate_text(
        ai_configuration, batched_summarization_messages, response_format=answer_schema.claim_summarization_format
    )
    content_items_list = []
    for content_items in summarized_claims:
        json_content_items = loads(content_items)
        for item in json_content_items['content_items']:
            content_items_list.append(item)
    content_items_dict = {i: v for i, v in enumerate(content_items_list)}
    content_items_context = dumps(content_items_dict, indent=2, ensure_ascii=False)
    content_item_messages = utils.prepare_messages(prompts.content_integration_prompt, {'content': content_items_context, 'question': question})
    content_structure = loads(utils.generate_text(ai_configuration, content_item_messages, response_format=answer_schema.content_integration_format))

    report, references, matched_chunks = build_report_markdown(
        question,
        content_items_dict,
        content_structure,
        processed_chunks.cid_to_text,
        source_to_supported_claims,
        source_to_contradicted_claims    
    )
    return AnswerObject(
        extended_answer=report,
        references=references,
        referenced_chunks=matched_chunks,
        net_new_sources=net_new_sources
    )

def build_report_markdown(question, content_items_dict, content_structure, cid_to_text, source_to_supported_claims, source_to_contradicted_claims):
    text_jsons = [loads(text) for text in cid_to_text.values()]
    matched_chunks = {f"{text['title']} ({text['chunk_id']})" : text for text in text_jsons}
    home_link = '#'+content_structure["report_title"].replace(' ', '-').lower()
    report = f'# Report\n\n## Question\n\n*{question}*\n\n## Answer\n\n{content_structure["answer"]}\n\n## Analysis\n\n### {content_structure["report_title"]}\n\n{content_structure["report_summary"]}\n\n'
    for theme in content_structure['theme_order']:
        report += f'#### Theme: {theme["theme_title"]}\n\n{theme["theme_summary"]}\n\n'
        for item_id in theme['content_id_order']:
            item = content_items_dict[item_id]
            report += f'##### {item["content_title"]}\n\n{item["content_summary"]}\n\n{item["content_commentary"]}\n\n'
        report += f'##### AI theme commentary\n\n{theme["theme_commentary"]}\n\n'
    report += f'#### AI report commentary\n\n{content_structure["report_commentary"]}\n\n'
    references = extract_chunk_references(report)
    print(report)
    print(references)
    report = link_chunk_references(report, references)
    report += f'## Sources\n\n'
    for ix, source_label in enumerate(references):
        if source_label in matched_chunks:
            supports_claims = source_to_supported_claims[source_label]
            contradicts_claims = source_to_contradicted_claims[source_label]
            supports_claims_str = '- ' + '\n- '.join([claim_statement for _, claim_statement in supports_claims])
            contradicts_claims_str = '- ' + '\n- '.join([claim_statement for _, claim_statement in contradicts_claims])
            report += f'#### Source {ix+1}\n\n<details>\n\n##### Text chunk: {source_label}\n\n{matched_chunks[source_label]["text_chunk"]}\n\n'
            if len(supports_claims) > 0:
                report += f'##### Supports claims\n\n{supports_claims_str}\n\n'
            if len(contradicts_claims) > 0:
                report += f'##### Contradicts claims\n\n{contradicts_claims_str}\n\n'
            report += f'</details>\n\n[Back to top]({home_link})\n\n'
        else:
            print(f'No match for {source_label}')
        
    return report, references, matched_chunks

def cluster_cids(
    relevant_cids,
    cid_to_vector,
    target_clusters
):
    clustered_cids = {}
    if len(relevant_cids) > 0:
        # use k-means clustering to group relevant cids into target_clusters clusters
        cids = [cid for cid in relevant_cids]
        vectors = [cid_to_vector[cid] for cid in cids]
        kmeans = cluster.KMeans(n_clusters=target_clusters)
        kmeans.fit(vectors)
        cluster_assignments = kmeans.predict(vectors)
        
        for i, cid in enumerate(cids):
            cluster_assignment = cluster_assignments[i]
            if cluster_assignment not in clustered_cids:
                clustered_cids[cluster_assignment] = []
            clustered_cids[cluster_assignment].append(cid)
    return clustered_cids

async def requery_claim(ai_configuration, question, units, neighbours, claim_embedding, claim_context, claim_statement, cid_to_text):
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
    chunks = dumps({i: cid_to_text[cid] for i, cid in enumerate(cids)}, ensure_ascii=False, indent=2)
    messages = utils.prepare_messages(prompts.claim_requery_prompt, {'question': question, 'claim': contextualized_claim, 'chunks': chunks})
    response = await utils.generate_text_async(ai_configuration, messages, response_format=answer_schema.claim_requery_format)
    chunk_titles = []
    for cid in cids:
        text_json = loads(cid_to_text[cid])
        chunk_titles.append(f"{text_json['title']} ({text_json['chunk_id']})")

    response_json = loads(response)
    supporting_sources = response_json['supporting_source_indicies']
    contradicting_sources = response_json['contradicting_source_indicies']
    supporting_source_labels = [chunk_titles[i] for i in supporting_sources]
    contradicting_source_labels = [chunk_titles[i] for i in contradicting_sources]
    return claim_context, claim_statement, supporting_source_labels, contradicting_source_labels