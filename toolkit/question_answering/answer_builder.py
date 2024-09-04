# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import re
from json import loads

import toolkit.AI.utils as utils
import toolkit.question_answering.helper_functions as helper_functions
import toolkit.question_answering.prompts as prompts


def extract_chunk_references(text):
    source_spans = re.finditer(r'\[source: (.+)\]', text, re.MULTILINE)
    references = set()
    for source_span in source_spans:
        parts = [x.strip() for x in source_span.group(1).split(',')]
        references.update(parts)
    return references

def convert_answer_object_to_text(answer_object):
    response = f'# {answer_object["title"]}\n\n'
    response += f'*In response to: {answer_object["question"]}*\n\n'
    response += f'## Introduction\n\n{answer_object["introduction"]}\n\n## Analysis\n\n'
    for item in answer_object['content_items']:
        response += f'### {item["title"]}\n\n{item["content"]}\n\n'
    response += f'## Conclusion\n\n{answer_object["conclusion"]}\n\n'
    return response

def update_answer_object(answer_object, answer_update):
    new_and_updated_ids = set([x['id'] for x in answer_update['content_items']])
    answer_object['title'] = answer_update['title']
    answer_object['introduction'] = answer_update['introduction']
    answer_object['content_id_sequence'] = answer_update['content_id_sequence']
    updated_content_items = []
    for item_id in answer_object['content_id_sequence']:
        if item_id in new_and_updated_ids:
            for item in answer_update['content_items']:
                if item['id'] == item_id:
                    updated_content_items.append(item)
                    break
        else:
            for item in answer_object['content_items']:
                if item['id'] == item_id:
                    updated_content_items.append(item)
                    break
    answer_object['content_items'] = updated_content_items
    answer_object['conclusion'] = answer_update['conclusion']

def generate_answer(
        ai_configuration,
        answer_object,
        answer_format,
        processing_queue,
        answer_batch_size,
        answer_stream,
        answer_callback
    ):
    selected_chunks = processing_queue[:answer_batch_size]
    for s in selected_chunks:
        processing_queue.remove(s)
    answer_messages = utils.prepare_messages(
        prompts.chunk_summarization_prompt, 
        {'chunks': selected_chunks, 'answer_object': answer_object}
    )
    answer_response = helper_functions.generate_text(ai_configuration, answer_messages, response_format=answer_format)
    update_answer_object(answer_object, loads(answer_response))
    answer_text = '\n\n'.join([x['content'] for x in answer_object['content_items']])
    references = extract_chunk_references(answer_text)
    answer_stream.insert(0, convert_answer_object_to_text(answer_object))
    if answer_callback is not None:
        answer_callback(answer_stream)
    return selected_chunks, references

def generate_answers(
        ai_configuration,
        answer_object,
        answer_format,
        process_chunks,
        answer_batch_size,
        answer_stream,
        answer_callback,
        answer_history,
        progress_callback
    ):
    all_selected_chunks = []
    remaining_chunks = list(set(process_chunks) - set(all_selected_chunks))
    while len(remaining_chunks) > 0:
        selected_chunks, references = generate_answer(
            ai_configuration,
            answer_object,
            answer_format,
            remaining_chunks,
            answer_batch_size,
            answer_stream,
            answer_callback
        )
        selected_metadata = set()
        for c in selected_chunks:
            try:
                c_json = loads(c)
                selected_metadata.add(f'{c_json["title"]} ({c_json["chunk_id"]})')
            except:
                selected_metadata.add("Unknown (Unknown)")

        used_references = [r for r in references if r in selected_metadata]
        answer_history.append((len(used_references), len(selected_chunks)))
        if progress_callback is not None:
            progress_callback(helper_functions.get_answer_progress(answer_history))
        all_selected_chunks.extend(selected_chunks)
        remaining_chunks = list(set(process_chunks) - set(all_selected_chunks))