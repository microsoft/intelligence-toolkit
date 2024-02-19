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

embedder = util.AI_API.create_embedder(cache='qa_mine\\embeddings') #SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
encoder = tiktoken.get_encoding('cl100k_base')

class Question:
    def __init__(self, file, text, vector, tier, id) -> None:
        self.file = file
        self.text = text
        self.id = id
        self.vector = vector
        self.tier = tier
        self.answer_texts = []
        self.answer_vectors = []
        self.super_qs = []
        self.sub_qs = []
        self.answer_references = []
        self.merged_questions = []

    def add_answer(self, answer, answer_references, qa_vector):
        self.answer_texts.append(answer)
        self.answer_references.append(answer_references)
        self.answer_vectors.append(qa_vector)

    def add_merged_question(self, q):
        if q not in self.merged_questions:
            self.merged_questions.append(q)

    def add_sub_q(self, sub_q):
        self.sub_qs.append(sub_q)

    def calculate_sub_q_spread(self):
        distances = []
        all_sub_qs = self.list_all_sub_questions()
        if len(all_sub_qs) > 0:
            for sq in all_sub_qs:
                distances.append(scipy.spatial.distance.cosine(self.vector, sq.vector))
            l = len(all_sub_qs)
            m = np.mean(distances)
            s = l*m
            return l, m, s
        else:
            return 0, 0.0, 0.0

    def list_all_sub_questions(self):
        sub_qs = []
        for sq in self.sub_qs:
            sub_qs.append(sq)
            sub_qs += sq.list_all_sub_questions()
        return sub_qs

    def generate_outline(self, level):
        outline = ''
        outline += level * '#' + f' Q{self.id}: {self.text}\n\n'
        if self.tier == 0:
            sources = ', '.join([f'{self.file.name} (p{p})' for (f, p) in self.answer_references[0]])
            outline += f'{self.answer_texts[0]} [sources: {sources}]\n\n'
            for mq in self.merged_questions:
                msources = ', '.join([f'{mq.file.name} (p{p})' for (f, p) in mq.answer_references[0]])
                outline += f'{mq.answer_texts[0]} [sources: {msources}]\n\n'
        merged_sqs = set(self.sub_qs)
        for mq in self.merged_questions:
            merged_sqs.update(mq.sub_qs)
        for sq in merged_sqs:
            outline += sq.generate_outline(level=level+1)
        return outline

    def __str__(self):
        return f'T{self.tier}Q{self.id}: {self.text}'
    
    def __repr__(self):
        return self.__str__()


class File:

    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id
        self.text = ''
        self.vector = []
        self.chunk_texts = {}
        self.chunk_vectors = {}
        # self.page_texts = {}
        # self.page_vectors = {}
        self.questions = {}

    def set_text(self, text):
        self.text = text

    def set_vector(self, vector):
        self.vector = vector

    # def add_page(self, page_text, page_vector, page_ix):
    #     self.page_texts[page_ix] = page_text
    #     self.page_vectors[page_ix] = page_vector

    def add_chunk(self, chunk_text, chunk_vector, chunk_id):
        self.chunk_texts[chunk_id] = chunk_text
        self.chunk_vectors[chunk_id] = chunk_vector

    def add_question(self, q):
        self.questions[q.id] = q

    def add_answer(self, qid, answer, references, qa_vector):
        self.questions[qid].add_answer(answer, references, qa_vector)

chunk_size = 5000
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
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
            file = File(file_link.name, file_id)
            sv.answering_files.value[file_id] = file
            bytes = file_link.getvalue()
            path = os.path.join('qa_mine\\raw_files', file.name)
            if not os.path.exists(path):
                with open(path, 'wb') as f:
                    f.write(bytes)
                pdf_reader = pdfplumber.open(io.BytesIO(bytes))
                for ix in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[ix].extract_text()
                    doc_text += f'\n[PAGE {ix+1}]\n\n{page_text}\n\n'
                file.set_text(doc_text)
                chunks = [x.strip() for x in text_splitter.split_text(doc_text)]
                paged_chunks = []
                for ix, chunk in enumerate(chunks):
                    page_matches = re.match(r'\[PAGE (\d+)\]', chunk)
                    if page_matches != None:
                        first_page = int(page_matches.groups()[0])
                        last_page = int(page_matches.groups()[-1])
                    if not chunk.startswith('[PAGE '):
                        chunk = f'[PAGE {last_page}]\n\n{chunk}'
                    chunk = f'[FILE {file.name}]\n\n{chunk}'
                    open(os.path.join('qa_mine\\text_chunks', f'{file.name[:-4]}-{ix+1}.txt'), 'wb').write(chunk.encode('utf-8'))
                    chunk_vec = embedder.encode(chunk)
                    paged_chunks.append(chunk)
                    file.add_chunk(chunk, chunk_vec, ix+1)                  
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

def mine_deeper_questions(sv):
    sv.answering_deeper_questions.value = {}
    num_tiers = sv.answering_max_tier.value
    for tier in range(0, num_tiers):
        cluster_to_qs = generate_question_clusters(sv, tier)
        cluster_placeholder = st.empty()
        for cx, (cluster, qs) in enumerate(cluster_to_qs.items()):
            if len(qs) > 0:
                qd = sv.answering_deeper_questions.value if tier > 0 else sv.answering_surface_questions.value
                q_texts = [f'{qid}: {qd[qid].text}' for qid in qs]
                q_text = '\n\n'.join(q_texts)
                cluster_placeholder.markdown(f'Mining cluster **{cx+1}** of **{len(cluster_to_qs)}** for tier **{tier+1}** of **{num_tiers}**:\n\n{q_text}')
            
                q_text_hash = hash(q_text)
                if os.path.exists(f'qa_mine\\questions\\{q_text_hash}.txt'):
                    print(f'{q_text_hash}.txt found. Reusing...')
                    deep_qas_raw = open(f'qa_mine\\questions\\{q_text_hash}.txt', 'r').read()
                else:
                    print(f'{q_text_hash}.txt not found. Generating...')
                    
                    deep_placeholder = st.empty()
                    system_message = """\
You are a helpful assistant extracting a single deeper question from a list of surface-level questions, each of which is prefixed by the question ID.

The identified deeper question should:

- give priority to the first question in the list, which is ranked by question importance
- represent deeper or higher-level questions that are answered by different combinations of input questions
- draw on as many input questions as possible to answer the deeper question, without generating redundant questions
- not duplicate any of the input questions

Further instructions:

Output the response as a JSON object, with the "question" field supported by a list of "sub_questions" IDs drawn from the input questions and formatted as a list of numbers.
Do not begin the response with ```json or end it with ```.
"""
                        
                    user_message = """\
Input questions:

{q_text}
"""
                    variables = {
                        'q_text': q_text
                    }
                    deep_qas_raw = util.AI_API.generate(
                        model=sv.model.value,
                        temperature=sv.temperature.value,
                        max_tokens=sv.max_tokens.value,
                        placeholder=deep_placeholder,
                        system_message=system_message,
                        user_message=user_message,
                        variables=variables,
                        prefix=''
                    )
                    deep_qas_raw = deep_qas_raw.replace('```json', '').replace('```', '').strip()
                    open(f'qa_mine\\questions\\{q_text_hash}.txt', 'w').write(deep_qas_raw)
                    deep_placeholder.empty()
                try:
                    with st.spinner('Parsing JSON...'):
                        x = json.loads(deep_qas_raw)
                        q = x['question']
                        qid = sv.answering_next_q_id.value
                        sv.answering_next_q_id.value += 1
                        sqxs = [int(y) for y in x['sub_questions']]
                        q_vec = embedder.encode(q)
                        new_q = Question(None, q, q_vec, tier+1, qid)
                        for sqx in sqxs:
                            sq = sv.answering_surface_questions.value[sqx] if tier == 0 else sv.answering_deeper_questions.value[sqx]
                            new_q.add_sub_q(sq)
                        sv.answering_deeper_questions.value[qid] = new_q
                except Exception as e:
                    print(e)
        cluster_placeholder.empty()

def mine_deeper_questions_from_question_list(sv, questions, placeholder, prefix):
    deep_qas_raw = ''
    if len(questions) > 0:
        q_texts = [f'{q.id}: {q.text}' for q in questions]
        q_text = '\n\n'.join(q_texts)    
        q_text_hash = hash(q_text)
        if os.path.exists(f'qa_mine\\questions\\{q_text_hash}.txt'):
            print(f'{q_text_hash}.txt found. Reusing...')
            deep_qas_raw = open(f'qa_mine\\questions\\{q_text_hash}.txt', 'r').read()
        else:
            print(f'{q_text_hash}.txt not found. Generating...')

            system_message = """\
You are a helpful assistant extracting deeper questions from a list of surface-level questions, each of which is prefixed by the question ID.

The identified deeper questions should:

- represent deeper or higher-level questions that are answered by the input questions
- not duplicate one another or any of the input questions

Further instructions:

Output the response as a JSON list, with the "question" field of each item supported by a list of "sub_questions" IDs drawn from the input questions and formatted as a list of numbers.
Do not begin the response with ```json or end it with ```.
"""
                
            user_message = """\
Input questions:

{q_text}
"""
            variables = {
                'q_text': q_text
            }
            deep_qas_raw = util.AI_API.generate(
                model=sv.model.value,
                temperature=sv.temperature.value,
                max_tokens=sv.max_tokens.value,
                placeholder=placeholder,
                system_message=system_message,
                user_message=user_message,
                variables=variables,
                prefix=prefix
            )
            deep_qas_raw = deep_qas_raw.replace('```json', '').replace('```', '').strip()
            open(f'qa_mine\\questions\\{q_text_hash}.txt', 'w').write(deep_qas_raw)
        try:
            with st.spinner('Parsing JSON...'):
                deep_qas = json.loads(deep_qas_raw)
                for x in deep_qas:
                    q = x['question']
                    qid = sv.answering_next_q_id.value
                    sv.answering_next_q_id.value += 1
                    sqxs = [int(y) for y in x['sub_questions']]
                    q_vec = embedder.encode(q)
                    new_q = Question(None, q, q_vec, 1, qid)
                    for sqx in sqxs:
                        sq = sv.answering_deeper_questions.value[sqx]
                        new_q.add_sub_q(sq)
                    sv.answering_deeper_questions.value[qid] = new_q
        except Exception as e:
            print(e)   
    return deep_qas_raw

def update_question(sv, question_history, new_questions, placeholder, prefix):
    response = question_history[-1]
    if len(new_questions) > 0:
        q_texts = [f'{q.id}: {q.text}\n\n{q.answer_texts[0]}' for q in new_questions]
        q_text = '\n\n'.join(q_texts)    
        

        system_message = """\
You are a helpful assistant augmenting a user question with any answers found in a list of input questions, each of which is prefixed by the question ID.

Any partial answers should be inserted in parentheses at the appropriate point in the question and reference the supporting question IDs using "[questions: Q<ID>, Q<ID>...]".

Do not insert any text indicating lack of knowledge, and do not remove any knowledge (including question references) already present in the previous augmented question.

Retain the structure of the original question, including any punctuation such as question marks. Do not add any new parts to the question other than the inserted answers.
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
        response = util.AI_API.generate(
            model=sv.model.value,
            temperature=sv.temperature.value,
            max_tokens=sv.max_tokens.value,
            placeholder=placeholder,
            system_message=system_message,
            user_message=user_message,
            variables=variables,
            prefix=prefix
        )
    else:
        print('Got no new questions!')
    return response

