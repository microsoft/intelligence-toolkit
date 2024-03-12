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

def generate_question_clusters_old(sv, tier):
    rows = []
    if tier == 0:
        for q in sv.answering_surface_questions.value.values():
            rows.append(['surface_q', q.id, q.vector])
    else:
        for q in sv.answering_deeper_questions.value.values():
            if q.tier == tier:
                rows.append(['deeper_q', q.id, q.vector])
    df = pd.DataFrame(rows, columns=['type', 'qid', 'vector'])
    if len(df) <= sv.answering_cluster_target.value:
        df['cluster'] = 0
        cluster_to_qs = {0: df['qid'].astype(int).tolist()}
    else:
        # cluster questions using DBSCAN
        epss = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        cluster_to_qs = {}
        for eps in epss:
            clustering = sklearn.cluster.DBSCAN(eps=eps, min_samples=1, metric='cosine').fit(df['vector'].tolist())
            df['cluster'] = clustering.labels_
            cluster_to_qs = {}
            for ix, row in df.iterrows():
                if row['cluster'] not in cluster_to_qs.keys():
                    cluster_to_qs[int(row['cluster'])] = []
                cluster_to_qs[int(row['cluster'])].append(int(row['qid']))
            max_cluster_size = max([len(qs) for qs in cluster_to_qs.values()])
            if max_cluster_size >= sv.answering_cluster_target.value:
                break
    # # remove singleton clusters
    # cluster_to_qs = {k: v for k, v in cluster_to_qs.items() if len(v) > 1}
    return cluster_to_qs

def generate_question_clusters(sv, tier):
    rows = []
    if tier == 0:
        for q in sv.answering_surface_questions.value.values():
            rows.append(['surface_q', q.id, q.vector])
    else:
        for q in sv.answering_deeper_questions.value.values():
            if q.tier == tier:
                rows.append(['deeper_q', q.id, q.vector])
    df = pd.DataFrame(rows, columns=['type', 'qid', 'vector'])
    # compute cosine distances between all pairs of questions
    qids = df['qid'].tolist()
    vectors = df['vector'].tolist()
    cluster_to_qs = {}
    if len(vectors) == 0:
        print('Nothing to cluster')
        return cluster_to_qs
    print(vectors)
    distances = scipy.spatial.distance.cdist(vectors, vectors, metric='cosine')
    k = 5
    limit = 3
    merge_dist = 0.05
    # for each question, find its k neareast neighbours
    
    q_to_ns = {}
    n_counter = Counter()
    for ix, qid in enumerate(qids):
        distances_to_q = distances[ix]
        nearest_ixs = np.argsort(distances_to_q)
        nearest_qids = [qids[i] for i in nearest_ixs]
        nearest_qids.remove(qid)
        q_to_ns[qid] = nearest_qids
        n_counter.update(nearest_qids)
        for nx in nearest_ixs:
            if nx == ix:
                continue
            nd = distances_to_q[nx]
            if nd < merge_dist:
                if tier == 0:
                    print(f'Merging {qid} and {qids[nx]}')
                    # print text
                    print(f'Q1: {sv.answering_surface_questions.value[qid].text}')
                    print(f'Q2: {sv.answering_surface_questions.value[qids[nx]].text}')
                    sv.answering_surface_questions.value[qid].add_merged_question(sv.answering_surface_questions.value[qids[nx]])
                    sv.answering_surface_questions.value[qids[nx]].add_merged_question(sv.answering_surface_questions.value[qid])
                else:
                    print(f'Merging {qid} and {qids[nx]}')
                    # print text
                    print(f'Q1: {sv.answering_deeper_questions.value[qid].text}')
                    print(f'Q2: {sv.answering_deeper_questions.value[qids[nx]].text}')
                    sv.answering_deeper_questions.value[qid].add_merged_question(sv.answering_deeper_questions.value[qids[nx]])
                    sv.answering_deeper_questions.value[qids[nx]].add_merged_question(sv.answering_deeper_questions.value[qid])
    freq_order = [x[0] for x in n_counter.most_common()]
    used_counter = Counter()
    for q in freq_order:
        if used_counter[q] < limit:
            usable_ns = [n for n in q_to_ns[q] if used_counter[n] < limit][:k]
            cluster_to_qs[q] = usable_ns
            used_counter.update(usable_ns)
    # remove singleton clusters
    cluster_to_qs = {k: v for k, v in cluster_to_qs.items() if len(v) > 1}
    return cluster_to_qs

def generate_data_context(sv, input_text, data_limit):
    qe = embedder.encode(input_text)
    all_qs = list(sv.answering_surface_questions.value.values()) + list(sv.answering_deeper_questions.value.values())
    cosine_distances = sorted([(q, scipy.spatial.distance.cosine(qe, q.vector)) for q in all_qs], key=lambda x:x[1], reverse=False)
    outline = ''
    used_qs = set()
    while len(encoder.encode(outline)) < data_limit:
        q = cosine_distances.pop(0)[0]
        if q.id in used_qs:
            continue
        used_qs.add(q.id)
        used_qs.update(q.list_all_sub_questions())
        to_add = q.generate_outline(level=1)
        if len(encoder.encode(outline + to_add)) < data_limit:
            outline += to_add
        else:
            break
    return outline


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

