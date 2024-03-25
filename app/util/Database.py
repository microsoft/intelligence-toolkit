import os
import duckdb

class Database:
    def __init__(self, cache, db_name) -> None:
        if not os.path.exists(cache):
            os.makedirs(cache)

        db_path = os.path.join(cache, f'{db_name}.db')
        self.connection = duckdb.connect(database=db_path)

    def create_table(self, name, attributes = []):
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(attributes)})")

    def select_embedding_from_hash(self, hash_text, username = ''):
        return self.connection.execute(f"SELECT embedding FROM embeddings WHERE hash_text = '{hash_text}' and username = '{username}'").fetchone()

    def insert_into_embeddings(self, hash_text, embedding, username = ''):
        self.connection.execute(f"INSERT INTO embeddings VALUES ('{username}','{hash_text}', {embedding})")

    def execute(self, query):
        return self.connection.execute(query)
    
    