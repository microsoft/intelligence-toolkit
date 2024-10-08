# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import re
from json import loads, dumps
import numpy as np

import toolkit.AI.utils as utils
import toolkit.query_text_data.helper_functions as helper_functions
import toolkit.query_text_data.answer_schema as answer_schema
import toolkit.query_text_data.prompts as prompts
import sklearn.cluster as cluster
import scipy.spatial

def extract_chunk_references(text):
    source_spans = re.finditer(r'\[source: (.+)\]', text, re.MULTILINE)
    references = set()
    for source_span in source_spans:
        parts = [x.strip() for x in source_span.group(1).split(',')]
        references.update(parts)
    return references

async def answer_question(
    ai_configuration,
    question,
    relevant_cids,
    cid_to_text,
    cid_to_vector,
    answer_batch_size,
    embedder,
    embedding_cache,
    select_logit_bias,
    claim_requery=False
):
    target_clusters = len(relevant_cids) // answer_batch_size
    if len(relevant_cids) / answer_batch_size > target_clusters:
        target_clusters += 1
    clustered_cids = cluster_cids(relevant_cids, cid_to_vector, target_clusters)
    print(clustered_cids)
    clustered_texts = [[cid_to_text[cid] for cid in cids] for cids in clustered_cids.values()]
    if claim_requery:
        batched_extraction_messages = [utils.prepare_messages(prompts.claim_extraction_prompt, {'chunks': texts, 'question': question}) 
                            for texts in clustered_texts]

        extracted_claims = await utils.map_generate_text(
            ai_configuration, batched_extraction_messages, response_format=answer_schema.claim_extraction_format
        )
        json_extracted_claims = [loads(claims) for claims in extracted_claims]
        def clean(c):
            cd = c.copy()
            for context in cd['claim_analysis']:
                for claim in context['claims']:
                    claim.pop('supporting_sources')
                    claim.pop('contradicting_sources')
            return cd
        cleaned_claims = [clean(c) for c in json_extracted_claims]
        
        for claim_sets in cleaned_claims:
            for claim_set in claim_sets['claim_analysis']:
                requery_claim(ai_configuration, claim_set, cid_to_text, cid_to_vector, embedder, embedding_cache, 5)

        batched_summarization_messages = [utils.prepare_messages(prompts.claim_summarization_prompt, {'analysis': claims, 'data': clustered_texts[i], 'question': question}) 
                            for i, claims in enumerate(extracted_claims)]
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

    report = build_report_markdown(question, content_items_dict, content_structure)
    references = extract_chunk_references(report)
    return report, references

def build_report_markdown(question, content_items_dict, content_structure):
    report = f'# {content_structure["report_title"]}\n\n*In response to: {question}*\n\n## Executive summary\n\n{content_structure["report_summary"]}\n\n'
    for theme in content_structure['theme_order']:
        report += f'## Theme: {theme["theme_title"]}\n\n{theme["theme_summary"]}\n\n'
        for item_id in theme['content_id_order']:
            item = content_items_dict[item_id]
            report += f'### {item["content_title"]}\n\n{item["content_summary"]}\n\n{item["content_commentary"]}\n\n'
        report += f'### AI theme commentary\n\n{theme["theme_commentary"]}\n\n'
    report += f'## AI report commentary\n\n{content_structure["report_commentary"]}\n\n'
    return report

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

def requery_claim(ai_configuration, claim_context, cid_to_text, cid_to_vector, embedder, embedding_cache, batch_size):
    print(f'Requerying claims: {dumps(claim_context, ensure_ascii=False, indent=2)}')
    claims = dumps(claim_context, ensure_ascii=False, indent=2)
    claim_embedding = np.array(
        embedder.embed_store_one(
            claims, embedding_cache
        )
    )
    all_units = sorted([(cid, vector) for cid, vector in (cid_to_vector.items())], key=lambda x: x[0])
    cosine_distances = sorted(
        [
            (cid, scipy.spatial.distance.cosine(claim_embedding, vector))
            for (cid, vector) in all_units
        ],
        key=lambda x: x[1],
        reverse=False,
    )
    # batch cids into batches of size batch_size
    batched_cids = [[cid for cid, dist in cosine_distances[i:i + batch_size]] for i in range(0, len(cosine_distances), batch_size)]
    for cid_batch in batched_cids:
        chunks = dumps({i: cid_to_text[cid] for i, cid in enumerate(cid_batch)}, ensure_ascii=False, indent=2)
        messages = utils.prepare_messages(prompts.claim_requery_prompt, {'claims': claims, 'chunks': chunks})
        response = loads(utils.generate_text(ai_configuration, messages, response_format=answer_schema.claim_requery_format))
        relevant_cids = set()
        chunk_titles = []
        for cid in cid_batch:
            text_json = loads(cid_to_text[cid])
            chunk_titles.append(f"{text_json['title']} ({text_json['chunk_id']})")
        for claim_analysis in response['claim_analysis']:
            claim_context_index = claim_analysis['claim_context_index']
            claim_statement_index = claim_analysis['claim_statement_index']
            supporting_sources = claim_analysis['supporting_source_indicies']
            contradicting_sources = claim_analysis['contradicting_source_indicies']
            supporting_source_labels = [chunk_titles[i] for i in supporting_sources]
            contradicting_source_labels = [chunk_titles[i] for i in contradicting_sources]
            claim_context['claim_analysis'][claim_context_index]['claims'][claim_statement_index]['supporting_sources'] = supporting_source_labels
            claim_context['claim_analysis'][claim_context_index]['claims'][claim_statement_index]['contradicting_sources'] = contradicting_source_labels
            relevant_cids.update(supporting_sources)
            relevant_cids.update(contradicting_sources)
        if len(relevant_cids) == 0:
            break