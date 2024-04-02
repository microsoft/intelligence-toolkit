# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import numpy as np
import tiktoken
import scipy.spatial.distance
import workflows.question_answering.config as config
import os


import util.Embedder

embedder = util.Embedder.create_embedder(cache=os.path.join(config.cache_dir,"qa_mine")) #SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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