# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import tiktoken
import numpy as np
from util.Database import Database
import util.session_variables
from util.openai_instance import _OpenAI

gen_model = 'gpt-4-turbo-preview'
embed_model = 'text-embedding-3-small'
text_encoder = 'cl100k_base'
max_gen_tokens = 4096
max_input_tokens = 128000
default_temperature = 0
max_embed_tokens = 8191

openai = _OpenAI()
encoder = tiktoken.get_encoding(text_encoder)

class Embedder:
    def __init__(self, cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens) -> None:
        sv = util.session_variables.SessionVariables('home')
        self.username = sv.username.value
        self.model = model
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
                final_embeddings[ix] = np.array(embeddings)
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
            embeddings = [x.embedding for x in openai.client().embeddings.create(input = batch_texts, model=self.model).data]
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
                embedding = openai.client().embeddings.create(input = [text], model=self.model).data[0].embedding
                if auto_save:
                    self.connection.insert_into_embeddings(hsh, embedding)
                return embedding
            except:
                print(f'Error embedding text: {text}')
                return None

def create_embedder(cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens):
    return Embedder(cache, model, encoder, max_tokens)