from openai import OpenAI
import tiktoken
import os
import numpy as np

gen_model = 'gpt-4-turbo-preview'
embed_model = 'text-embedding-3-small'
text_encoder = 'cl100k_base'
max_gen_tokens = 4096
max_input_tokens = 128000
default_temperature = 0
max_embed_tokens = 8191
import duckdb

client = OpenAI()
encoder = tiktoken.get_encoding(text_encoder)

class Embedder:
    def __init__(self, cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens) -> None:
        self.model = model
        self.encoder = tiktoken.get_encoding(encoder)
        self.max_tokens = max_tokens
        if not os.path.exists(cache):
            os.makedirs(cache)
        self.connection = duckdb.connect(database=f'{cache}\\embeddings.db')
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings (hash_text STRING, embedding DOUBLE[])")
        

    def encode_all(self, texts):
        final_embeddings = [None] * len(texts)
        new_texts  = []
        for ix, text in enumerate(texts):
            text = text.replace("\n", " ")
            hsh = hash(text)
            exists = self.connection.execute(f"SELECT embedding FROM embeddings WHERE hash_text = '{hsh}'").fetchone()
            if not exists:
                new_texts.append((ix, text))
            else:
                final_embeddings[ix] = np.array(exists)
        print(f'Got {len(new_texts)} new texts')
        # split into batches of 2000
        for i in range(0, len(new_texts), 2000):
            batch = new_texts[i:i+2000]
            batch_texts = [x[1] for x in batch]
            embeddings = [x.embedding for x in client.embeddings.create(input = batch_texts, model=self.model).data]
            for j, (ix, text) in enumerate(batch):
                hsh = hash(text)
                self.connection.execute(f"INSERT INTO embeddings VALUES ('{hsh}', {embeddings[j]})")
                final_embeddings[ix] = np.array(embeddings[j])
        return np.array(final_embeddings)

    def encode(self, text):
        text = text.replace("\n", " ")
        hsh = hash(text)
        exists = self.connection.execute(f"SELECT embedding FROM embeddings WHERE hash_text = '{hsh}'").fetchone()

        if exists:
            return np.array(exists[0])
            # return [float(x) for x in open(path, 'r').read().split('\n') if len(x) > 0]
        else:
            tokens = len(self.encoder.encode(text))
            if tokens > self.max_tokens:
                text = text[:self.max_tokens]
                print('Truncated text to max tokens')
            try:
                embedding = client.embeddings.create(input = [text], model=self.model).data[0].embedding
                self.connection.execute(f"INSERT INTO embeddings VALUES ('{hsh}', {embedding})")
                return np.array(embedding)
            except:
                print(f'Error embedding text: {text}')
                return None

def create_embedder(cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens):
    return Embedder(cache, model, encoder, max_tokens)