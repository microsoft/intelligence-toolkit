import json
from openai import OpenAI
import tiktoken
import numpy as np
import duckdb

gen_model = 'gpt-4-turbo-preview'
embed_model = 'text-embedding-3-small'
text_encoder = 'cl100k_base'
max_gen_tokens = 4096
max_input_tokens = 128000
default_temperature = 0
max_embed_tokens = 8191

client = OpenAI()
encoder = tiktoken.get_encoding(text_encoder)

class Embedder1:
    def __init__(self, cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens):
        self.model = model
        self.encoder = tiktoken.get_encoding(encoder)
        self.max_tokens = max_tokens
        self.cache = cache

        # Connect to DuckDB database
        self.connection = duckdb.connect(database=cache+'11.db')

        # Create a table to store embeddings if it doesn't exist
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings (text_hash STRING, embedding FLOAT)")

    # def __del__(self):
    #     self.connection.close()

    def encode_all(self, texts):
        final_embeddings = [None] * len(texts)
        new_texts  = []
        for ix, text in enumerate(texts):
            text = text.replace("\n", " ")
            hsh = hash(text)
            # Check if embedding exists in DuckDB
            result = self.connection.execute(f"SELECT embedding FROM embeddings WHERE text_hash = '{hsh}'").fetchall()
            if result:
                final_embeddings[ix] = np.array(result)
            else:
                new_texts.append((ix, text))

        print(f'Got {len(new_texts)} new texts')
        # split into batches of 2000
        for i in range(0, len(new_texts), 2000):
            batch = new_texts[i:i+2000]
            batch_texts = [x[1] for x in batch]
            embeddings = [x.embedding for x in client.embeddings.create(input = batch_texts, model=self.model).data]

            for j, (ix, text) in enumerate(batch):
                hsh = hash(text)
                # Insert new embeddings into DuckDB
                for embedding in embeddings[j]:
                    self.connection.execute(f"INSERT INTO embeddings VALUES ('{hsh}', {embedding})")
                # self.connection.execute(f"INSERT INTO embeddings VALUES ('{hsh}', {embeddings[j]})")
                final_embeddings[ix] = np.array(embeddings[j])
        print('final_embeddingsfinal_embeddingsfinal_embeddings', final_embeddings)
        return np.array(final_embeddings)

    def encode(self, text):
        text = text.replace("\n", " ")
        hsh = hash(text)
        # Check if embedding exists in DuckDB
        result = self.connection.execute(f"SELECT embedding FROM embeddings WHERE text_hash = '{hsh}'").fetchall()
        if result:
            return np.array(result)
        else:
            tokens = len(self.encoder.encode(text))
            if tokens > self.max_tokens:
                text = text[:self.max_tokens]
                print('Truncated text to max tokens')
            try:
                embedding = client.embeddings.create(input=[text], model=self.model).data[0].embedding
                # Insert new embedding into DuckDB
                for emb in embedding:
                    self.connection.execute(f"INSERT INTO embeddings VALUES ('{hsh}', {emb})")
                print('final_embeddingsfinal_embeddingsfinal_embeddings', embedding)
                return np.array(embedding)
            except:
                print(f'Error embedding text: {text}')
                return None


def create_embedder(cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens):
    return Embedder1(cache, model, encoder, max_tokens)