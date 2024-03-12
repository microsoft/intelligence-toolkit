import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
import os
import sklearn.cluster
import json
import io
import tiktoken
import pdfplumber
import scipy.spatial.distance
from langchain.text_splitter import RecursiveCharacterTextSplitter

import util.AI_API
import workflows.question_answering.classes as classes
import workflows.question_answering.config as config

embedder = util.AI_API.create_embedder(cache='qa_mine\\embeddings')
encoder = tiktoken.get_encoding('cl100k_base')


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    length_function=len,
    is_separator_regex=False,
)

def chunk_files(sv, files):
    pb = st.progress(0, 'Chunking and embedding files...')
    for fx, file_link in enumerate(files):
        pb.progress((fx+1) / len(files), f'Chunking and embedding file {fx+1} of {len(files)}...')
        file_names = [f.name for f in sv.answering_files.value.values()]
        doc_text = ''
        if file_link.name not in file_names:
            file_id = sv.answering_next_file_id.value
            sv.answering_next_file_id.value += 1
            file = classes.File(file_link.name, file_id)
            sv.answering_files.value[file_id] = file
            bytes = file_link.getvalue()
            path = os.path.join('qa_mine\\raw_files', file.name)
            if not os.path.exists(path):
                with open(path, 'wb') as f:
                    f.write(bytes)
                pdf_reader = pdfplumber.open(io.BytesIO(bytes))
                cx = 1
                for px in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[px].extract_text()
                    doc_text += f'\n[PAGE {px+1}]\n\n{page_text}\n\n'
                    chunks = [x.strip() for x in text_splitter.split_text(page_text)]
                    paged_chunks = []
                    for ix, chunk in enumerate(chunks):
                        chunk = f'[PAGE {px+1}]\n\n{chunk}'
                        chunk = f'[FILE {file.name}]\n\n{chunk}'
                        open(os.path.join('qa_mine\\text_chunks', f'{file.name[:-4]}-p{px+1}-c{cx}.txt'), 'wb').write(chunk.encode('utf-8'))
                        chunk_vec = embedder.encode(chunk)
                        paged_chunks.append(chunk)
                        file.add_chunk(chunk, chunk_vec, cx)
                        cx += 1
                file.set_text(doc_text)
                                  
                # doc_vector = embedder.encode(doc_text, normalize_embeddings=True)
                # file.set_vector(doc_vector)
                with open(os.path.join('qa_mine\\text_files', file.name+'.txt'), 'wb') as f:
                    f.write(doc_text.encode('utf-8'))
            else:
                doc_text = open(os.path.join('qa_mine\\text_files', file.name+'.txt'), 'r', encoding='utf-8', errors='ignore').read()
                file.set_text(doc_text)
                cx = 1
                while os.path.exists(os.path.join('qa_mine\\text_chunks', f'{file.name[:-4]}-{cx}.txt')):
                    chunk = open(os.path.join('qa_mine\\text_chunks', f'{file.name[:-4]}-{cx}.txt'), 'r', encoding='utf-8', errors='ignore').read()
                    chunk_vec = embedder.encode(chunk)
                    file.add_chunk(chunk, chunk_vec, cx)
                    cx += 1
                
    pb.empty()

def update_question(sv, question_history, new_questions, placeholder, prefix):
    response = question_history[-1]
    if len(new_questions) > 0:
        q_texts = [f'{q.id}: {q.text}\n\n{q.answer_texts[0]}' for q in new_questions]
        q_text = '\n\n'.join(q_texts)    
        

        system_message = """\
You are a helpful assistant augmenting a user question with any relevant keywords (e.g., entities, concepts, or knowledge) found in a list of input questions, each of which is prefixed by the question ID.

Any relevant keywords should be inserted as a list, enclosed by parentheses, at the appropriate point in the question, with each keywords item referencing the supporting question IDs using "(<keywords 1> [Q<ID>, Q<ID>...], <keywords 2> [Q<ID>, Q<ID>...], ...)".

Do not insert any text indicating lack of relevant keywords, and do not remove any text (including question references) already present in the previous augmented question unless it is clearly irrelevant.

Retain the structure of the original question, including any punctuation such as question marks. Do not add any new parts to the question other than the inserted keywords.
"""
                
        user_message = """\
Original question: {original_question}

Previous augmented question: {augmented_question}

Input questions:

{q_text}

New augmented question adding to the prevous augmented question:
"""
        variables = {
            'original_question': question_history[0],
            'augmented_question': question_history[-1],
            'q_text': q_text
        }
        messages = util.AI_API.prepare_messages_from_message_pair(
            system_message=system_message,
            user_message=user_message,
            variables=variables,
        )
        response = util.AI_API.generate_text_from_message_list(
            messages=messages,
            placeholder=placeholder,
            prefix=prefix
        )
    else:
        print('Got no new questions!')
    return response

