from openai import OpenAI
import tiktoken
import os
import json
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
connection = duckdb.connect(database='cache\\question_answering\\embeddings.db')
connection.execute("CREATE TABLE IF NOT EXISTS qa_mine (hash_text STRING, embedding DOUBLE[])")
connection.execute("CREATE TABLE IF NOT EXISTS embeddings (hash_text STRING, embedding DOUBLE[])")
connection.execute("CREATE TABLE IF NOT EXISTS record_matching (hash_text STRING, embedding DOUBLE[])")
connection.execute("CREATE TABLE IF NOT EXISTS risk_networks (hash_text STRING, embedding DOUBLE[])")

def prepare_messages_from_message(system_message, variables):                
    messages = [
        {
            'role': 'system',
            'content': system_message.format(**variables)
        },
    ]
    return messages

def prepare_messages_from_message_pair(system_message, user_message, variables):                
    messages = [
        {
            'role': 'system',
            'content': system_message.format(**variables)
        },
        {
            'role': 'user',
            'content': user_message.format(**variables)
        }
    ]
    return messages

def count_tokens_in_message_list(messages):
    return len(encoder.encode(json.dumps(messages)))

def generate_text_from_message_list(messages, placeholder=None, prefix='', model=gen_model, temperature=default_temperature, max_tokens=max_gen_tokens):     
    response = ''
    try:
        responses = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            stream=True
        )
        response = ''             
        for chunk in responses:
            diff = chunk.choices[0].delta.content # type: ignore
            if diff is not None:
                response += diff
                if placeholder is not None:
                    show = prefix + response
                    if len(diff) > 0:
                        show += '▌'
                    placeholder.markdown(show, unsafe_allow_html=True)
        if placeholder is not None:
            placeholder.markdown(prefix + response, unsafe_allow_html=True)
    except Exception as e:
        print(f'Error generating from message list: {e}')
    return response

class Embedder:
    def __init__(self, cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens) -> None:
        self.model = model
        self.encoder = tiktoken.get_encoding(encoder)
        self.max_tokens = max_tokens
        self.cache = cache
        if not os.path.exists(cache):
            os.makedirs(cache)
        

    def encode_all(self, texts, table):
        final_embeddings = [None] * len(texts)
        new_texts  = []
        for ix, text in enumerate(texts):
            text = text.replace("\n", " ")
            hsh = hash(text)
            exists = connection.execute(f"SELECT embedding FROM {table} WHERE hash_text = '{hsh}'").fetchone()
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
                connection.execute(f"INSERT INTO {table} VALUES ('{hsh}', {embeddings[j]})")
                final_embeddings[ix] = np.array(embeddings[j])
        return np.array(final_embeddings)

    def encode(self, text, table):
        text = text.replace("\n", " ")
        hsh = hash(text)
        exists = connection.execute(f"SELECT embedding FROM {table} WHERE hash_text = '{hsh}'").fetchone()

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
                connection.execute(f"INSERT INTO {table} VALUES ('{hsh}', {embedding})")
                return np.array(embedding)
            except:
                print(f'Error embedding text: {text}')
                return None

def create_embedder(cache, model=embed_model, encoder=text_encoder, max_tokens=max_embed_tokens):
    return Embedder(cache, model, encoder, max_tokens)