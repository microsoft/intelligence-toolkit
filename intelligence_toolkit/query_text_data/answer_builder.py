# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import re
from json import loads, dumps
from collections import defaultdict

import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.answer_schema as answer_schema
import intelligence_toolkit.query_text_data.prompts as prompts
from intelligence_toolkit.query_text_data.classes import AnswerObject
import sklearn.cluster as cluster


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

async def answer_query(
    ai_configuration,
    query,
    expanded_query,
    processed_chunks,
    clustered_cids,
    cid_to_vector,
    target_chunks_per_cluster
):
    print(f"Answering query with clustered ids: {clustered_cids}")
    partitioned_texts = {}
    for theme, cids in clustered_cids.items():
        if len(cids) > target_chunks_per_cluster:
            cluster_to_cids = cluster_cids(cids, cid_to_vector, len(cids) // target_chunks_per_cluster)
            for cluster, cids in cluster_to_cids.items():
                partitioned_texts[f"{theme} - topic {cluster}"] = [f"{cid}: {processed_chunks.cid_to_text[cid]}" for cid in cids]
        else:
            partitioned_texts[theme] = [f"{cid}: {processed_chunks.cid_to_text[cid]}" for cid in cids]
    net_new_sources = 0
    batched_summarization_messages = [
        utils.prepare_messages(
            prompts.theme_summarization_prompt,
            {"chunks": texts, "theme": theme, "query": expanded_query},
        )
        for theme, texts in partitioned_texts.items()
    ]

    summarized_themes = await utils.map_generate_text(
        ai_configuration,
        batched_summarization_messages,
        response_format=answer_schema.theme_summarization_format,
    )

    theme_integration_messages = utils.prepare_messages(
        prompts.theme_integration_prompt,
        {"content": summarized_themes, "query": query},
    )

    report_wrapper = utils.generate_text(
        ai_configuration,
        theme_integration_messages,
        response_format=answer_schema.theme_integration_format,
    )

    report, references, matched_chunks = build_report_markdown(
        query,
        expanded_query,
        summarized_themes,
        report_wrapper,
        processed_chunks.cid_to_text
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
    summarized_themes,
    report_wrapper,
    cid_to_text
):
    summarized_themes_objs = [loads(text) for text in summarized_themes]
    report_wrapper_obj = loads(report_wrapper)
    text_jsons = [loads(text) for text in cid_to_text.values()]
    matched_chunks = {
        f"{text['title']} ({text['chunk_id']})": text for text in text_jsons
    }
    home_link = "#final-report"
    report = f'## Query\n\n*{query}*\n\n## Expanded Query\n\n*{expanded_query}*\n\n## Answer\n\n{report_wrapper_obj["answer"]}\n\n## Analysis\n\n### {report_wrapper_obj["report_title"]}\n\n{report_wrapper_obj["report_overview"]}\n\n'
    for theme in summarized_themes_objs:
        report += f'#### Theme: {theme["theme_title"]}\n\n'
        for point in theme["theme_points"]:
            report += f'##### {point["point_title"]}\n\n{point["point_evidence"]}\n\n{point["point_commentary"]}\n\n'
    report += (
        f'#### Implications\n\n{report_wrapper_obj["report_implications"]}\n\n'
    )
    report, references = extract_and_link_chunk_references(report)
    print(f"Extracted references: {references}")
    report += f"## Sources\n\n"
    for cid in references:
        if cid in cid_to_text.keys():
            chunk = loads(cid_to_text[cid])
            report += f'#### Source {cid}\n\n<details>\n\n##### Text chunk: {chunk["title"]} ({chunk["chunk_id"]})\n\n{chunk["text_chunk"]}\n\n'
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


