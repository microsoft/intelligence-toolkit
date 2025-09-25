# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.

import math
import re
from json import loads

import intelligence_toolkit.AI.utils as utils
import intelligence_toolkit.query_text_data.answer_schema as answer_schema
import intelligence_toolkit.query_text_data.prompts as prompts
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


def select_representative_cids(cids, cid_to_vector, target_count, min_retention_ratio):
    min_retention_ratio = max(0.0, min(1.0, min_retention_ratio))
    if target_count <= 0 and min_retention_ratio <= 0:
        return []

    minimum_count = math.ceil(len(cids) * min_retention_ratio)
    effective_target = max(target_count, minimum_count)
    if len(cids) <= effective_target:
        return list(cids)

    vectorized_cids = [
        (cid, cid_to_vector[cid])
        for cid in cids
        if cid in cid_to_vector
    ]

    if not vectorized_cids:
        return list(cids)[:effective_target]

    centroid = _mean_vector([vector for _, vector in vectorized_cids])
    distances = {
        cid: _squared_distance(vector, centroid)
        for cid, vector in vectorized_cids
    }

    ranked_cids = sorted(distances.keys(), key=lambda cid: distances[cid])
    selected_vector_cids = set(ranked_cids[:effective_target])

    selected_cids = []
    for cid in cids:
        if cid in selected_vector_cids and cid not in selected_cids:
            selected_cids.append(cid)
        if len(selected_cids) == effective_target:
            break

    if len(selected_cids) < effective_target:
        for cid in cids:
            if cid not in selected_cids:
                selected_cids.append(cid)
            if len(selected_cids) == effective_target:
                break

    return selected_cids


def _mean_vector(vectors):
    if not vectors:
        return []

    vector_length = len(vectors[0])
    sums = [0.0] * vector_length
    for vector in vectors:
        for i, value in enumerate(vector):
            sums[i] += value
    return [value / len(vectors) for value in sums]


def _squared_distance(vector_a, vector_b):
    return sum((a - b) ** 2 for a, b in zip(vector_a, vector_b))

async def answer_query(
    ai_configuration,
    query,
    expanded_query,
    processed_chunks,
    clustered_cids,
    cid_to_vector,
    max_chunks_per_theme,
    min_chunk_retention_ratio=0.6,
):
    print(f"Answering query with clustered ids: {clustered_cids}")
    partitioned_texts = {}
    chunk_cap = max(1, 10)
    for theme, cids in clustered_cids.items():
        selected_cids = select_representative_cids(
            cids,
            cid_to_vector,
            chunk_cap,
            min_chunk_retention_ratio,
        )
        if len(selected_cids) < len(cids):
            print(
                "Pruned theme '%s' from %d to %d representative chunks" % (
                    theme,
                    len(cids),
                    len(selected_cids),
                )
            )
        partitioned_texts[theme] = [
            f"{cid}: {processed_chunks.cid_to_text[cid]}"
            for cid in selected_cids
            if cid in processed_chunks.cid_to_text
        ]
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
        processed_chunks.cid_to_text,
        ai_configuration
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
    cid_to_text,
    ai_configuration
):
    summarized_themes_objs = [loads(text) for text in summarized_themes]
    report_wrapper_obj = loads(report_wrapper)
    text_jsons = [loads(text) for text in cid_to_text.values()]
    matched_chunks = {
        f"{text['title']} ({text['chunk_id']})": text for text in text_jsons
    }
    home_link = "#final-report"
    report = f'## AQuery\n\n*{query}*\n\n## Expanded Query\n\n*{expanded_query}*\n\n## Answer\n\n{report_wrapper_obj["answer"]}\n\n## Analysis\n\n### {report_wrapper_obj["report_title"]}\n\n{report_wrapper_obj["report_overview"]}\n\n'
    
    # Deduplicate and consolidate themes to avoid repetition
    themes_deduplication_prompt = """
    You are given a list of theme summaries that may contain overlapping or duplicate information. 
    Your task is to consolidate these themes into concise summaries without duplication.
    Check each theme and if their texts are too similar, merge them into one theme.
    
    - Maintain source references in the format [source: X]
    - Avoid repeating the same information across themes
    - Keep summaries concise and focused on main insights only
    
    Themes to consolidate: {themes}
    
    Please when creating new theme item, check if source references wheren't used in another theme, so don't use them again.
    Make sure to keep track of all source references used across themes.
    Don't repeat themes, so it's ok to return less themes than it was given you.
    Return consolidated themes as a JSON array where each theme has:
    - theme_title: Clear, non-overlapping title
    - theme_summary: Brief summary of key points with source references.
    """
    
    themes_json = [theme for theme in summarized_themes_objs]
    deduplication_messages = utils.prepare_messages(
        themes_deduplication_prompt,
        {"themes": themes_json}
    )
    
    # Note: This would need to be async in the actual implementation
    # For now, using the original themes if deduplication fails
    try:
        deduplicated_themes_response = utils.generate_text(
            ai_configuration,  # This parameter needs to be passed to this function
            deduplication_messages,
            response_format={"type": "json_object"}
        )
        deduplicated_themes = loads(deduplicated_themes_response)
        if isinstance(deduplicated_themes, list):
            summarized_themes_objs = deduplicated_themes
    except Exception as e:
        print(f"Theme deduplication failed, using original themes: {e}")
    
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


