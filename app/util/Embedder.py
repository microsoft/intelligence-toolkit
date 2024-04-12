# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import tiktoken
import numpy as np
from util.AI_API import generate_embedding_from_text
from util.Database import Database
import util.session_variables
from util.openai_instance import _OpenAI
import streamlit as st

text_encoder = 'cl100k_base'
max_gen_tokens = 4096
max_input_tokens = 128000
default_temperature = 0
max_embed_tokens = 8191

openai = _OpenAI()
encoder = tiktoken.get_encoding(text_encoder)

class Embedder:
    def __init__(self, cache, encoder=text_encoder, max_tokens=max_embed_tokens) -> None:
        sv = util.session_variables.SessionVariables('home')
        self.username = sv.username.value
        self.encoder = tiktoken.get_encoding(encoder)
        self.max_tokens = max_tokens
        self.connection = Database(cache, 'embeddings')
        self.connection.create_table('embeddings', ['username STRING','hash_text STRING UNIQUE', 'embedding DOUBLE[]'])

    def encode_all(self, texts):
        final_embeddings = [None] * len(texts)
        new_texts  = []
        for ix, text in enumerate(texts):
            text = text.replace("\n", " ")
            hsh = hash(text)
            embeddings = self.connection.select_embedding_from_hash(hsh)
            if not embeddings:
                new_texts.append((ix, text))
            else:
                final_embeddings[ix] = np.array(embeddings[0])
        print(f'Got {len(new_texts)} new texts')
        # split into batches of 2000
        pb = st.progress(0, 'Embedding text batches...')
        num_batches = len(new_texts) // 2000 + 1
        bi = 1
        for i in range(0, len(new_texts), 2000):
            pb.progress((bi) / num_batches, f'Embedding text batch {bi} of {num_batches}...')
            bi += 1
            batch = new_texts[i:i+2000]
            batch_texts = [x[1] for x in batch]
            list_all_embeddings = []
            try:
                embeddings = [x.embedding for x in generate_embedding_from_text(batch_texts).data]
            except Exception as e:
                raise Exception(f'Problem in OpenAI response. {e}')
                
            for j, (ix, text) in enumerate(batch):
                hsh = hash(text)
                list_all_embeddings.append((hsh, embeddings[j]))
                final_embeddings[ix] = np.array(embeddings[j])
            self.connection.insert_multiple_into_embeddings(list_all_embeddings) 
        pb.empty()
        return np.array(final_embeddings)

    def encode(self, text, auto_save = True):
        text = text.replace("\n", " ")
        hsh = hash(text)
        embeddings = self.connection.select_embedding_from_hash(hsh)

        if embeddings:
            return embeddings[0]
        else:
            tokens = len(self.encoder.encode(text))
            if tokens > self.max_tokens:
                text = text[:self.max_tokens]
                print('Truncated text to max tokens')
            try:
                embedding = generate_embedding_from_text([text]).data[0].embedding
            except Exception as e:
                raise Exception(f'Problem in OpenAI response. {e}')
                
            if auto_save:
                self.connection.insert_into_embeddings(hsh, embedding, self.username)
            return embedding

def create_embedder(cache, encoder=text_encoder, max_tokens=max_embed_tokens):
    return Embedder(cache, encoder, max_tokens)