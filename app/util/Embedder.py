# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import hashlib
import os
import pickle
import tiktoken
import numpy as np
from util.constants import MAX_SIZE_EMBEDDINGS_KEY, EMBEDDINGS_FILE_NAME
from util.SecretsHandler import SecretsHandler
from util.AI_API import generate_embedding_from_text
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
        if not os.path.exists(cache):
            os.makedirs(cache)
        self.file_path = os.path.join(cache, EMBEDDINGS_FILE_NAME)
        self.secrets_handler = SecretsHandler()

    def encode_all(self, texts):
        final_embeddings = [None] * len(texts)
        new_texts  = []
        loaded_embeddings = self.return_embeddings_list()
        for ix, text in enumerate(texts):
            text = text.replace("\n", " ")
            hsh = hashlib.sha256(text.encode()).hexdigest()
            embedding = self.return_existing_embedding(hsh, loaded_embeddings)
            if not len(embedding):
                new_texts.append((ix, text))
            else:
                final_embeddings[ix] = np.array(embedding)
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
            try:
                embeddings = [x.embedding for x in generate_embedding_from_text(batch_texts).data]
            except Exception as e:
                raise Exception(f'Problem in OpenAI response. {e}')
                
            for j, (ix, text) in enumerate(batch):
                hsh = hash(text)
                loaded_embeddings.update({hsh: embeddings[j]})
                final_embeddings[ix] = np.array(embeddings[j])
        self.save_embeddings_list(loaded_embeddings)
        pb.empty()
        return np.array(final_embeddings)

    def encode(self, text):
        text = text.replace("\n", " ")
        hsh = hashlib.sha256(text.encode()).hexdigest()
        loaded_embeddings = self.return_embeddings_list()
        embedding = self.return_existing_embedding(hsh, loaded_embeddings)

        if not embedding:
            tokens = len(self.encoder.encode(text))
            if tokens > self.max_tokens:
                text = text[:self.max_tokens]
                print('Truncated text to max tokens')
            try:
                embedding = generate_embedding_from_text([text]).data[0].embedding
            except Exception as e:
                raise Exception(f'Problem in OpenAI response. {e}')
                
            loaded_embeddings.update({hsh: embedding})
            self.save_embeddings_list(loaded_embeddings)

        return embedding

    def return_embeddings_list(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_embeddings_list(self, embeddings_list):
        embeddings_count = len(embeddings_list)
        max_size = int(self.secrets_handler.get_secret(MAX_SIZE_EMBEDDINGS_KEY)) or 0
        if max_size > 0 and embeddings_count > max_size:
            embeddings_list = dict(list(embeddings_list.items())[-max_size:])

        with open(self.file_path, '+wb') as f:
            pickle.dump(embeddings_list, f)
    
    def reset_embeddings(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def return_existing_embedding(self, hsh, embeddings_list):
        if hsh in embeddings_list:
            return embeddings_list[hsh]
        return {}

def create_embedder(cache, encoder=text_encoder, max_tokens=max_embed_tokens):
    return Embedder(cache, encoder, max_tokens)

