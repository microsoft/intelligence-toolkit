# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import logging

import numpy as np

from python.AI.defaults import DEFAULT_LOCAL_EMBEDDING_MODEL

from .cache_pickle import CachePickle
from .client import OpenAIClient
from .openai_configuration import OpenAIConfiguration
from .utils import get_token_count, hash_text

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self, configuration: OpenAIConfiguration | None, pickle_path=None, local=False
    ) -> None:
        self.configuration = configuration or OpenAIConfiguration()
        self.openai_client = OpenAIClient(configuration)
        self.pickle = CachePickle(path=pickle_path)
        self.local_client = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
        self.local = local

    def embed_store_one(self, text: str, cache_data=True):
        text_hashed = hash_text(text)
        loaded_embeddings = self.pickle.get_all() if cache_data else {}
        embedding = (
            self.pickle.get(text_hashed, loaded_embeddings) if cache_data else {}
        )
        if not embedding:
            tokens = get_token_count(text)
            if tokens > self.configuration.max_tokens:
                text = text[: self.configuration.max_tokens]
                logger.info("Truncated text to max tokens")
            try:
                if self.local:
                    embedding = self.local_client.encode(text).tolist()
                else:
                    embedding = self.openai_client.generate_embedding([text])
            except Exception as e:
                msg = f"Problem in OpenAI response. {e}"
                raise Exception(msg)
            loaded_embeddings.update({text_hashed: embedding})
            self.pickle.save(loaded_embeddings) if cache_data else None
        return embedding

    def embed_store_many(self, texts: list[str], callback=None, cache_data=True):
        final_embeddings = [None] * len(texts)
        new_texts = []
        loaded_embeddings = self.pickle.get_all() if cache_data else {}
        existing_texts_count = 0
        for ix, text in enumerate(texts):
            text_hashed = hash_text(text)
            embedding = (
                self.pickle.get(text_hashed, loaded_embeddings) if cache_data else {}
            )
            if not len(embedding):
                new_texts.append((ix, text))
            else:
                final_embeddings[ix] = np.array(embedding)
                existing_texts_count += 1
        print(f"Got {existing_texts_count} existing texts")
        logger.info("Got %s existing texts", existing_texts_count)
        print(f"Got {len(new_texts)} new texts")
        logger.info("Got %s new texts", len(new_texts))

        num_batches = len(new_texts) // 2000 + 1
        batch_count = 1
        for i in range(0, len(new_texts), 2000):
            if callback:
                for cb in callback:
                    cb.on_embedding_batch_change(batch_count, num_batches)
            batch_count += 1
            batch = new_texts[i : i + 2000]
            batch_texts = [x[1] for x in batch]
            try:
                if self.local:
                    embeddings = self.local_client.encode(batch_texts).tolist()
                else:
                    embeddings = [
                        x.embedding
                        for x in self.openai_client.generate_embeddings(
                            batch_texts
                        ).data
                    ]
            except Exception as e:
                msg = f"Problem in OpenAI response. {e}"
                raise Exception(msg)

            for j, (ix, text) in enumerate(batch):
                text_hashed = hash_text(text)
                loaded_embeddings.update({text_hashed: embeddings[j]})
                final_embeddings[ix] = np.array(embeddings[j])
        self.pickle.save(loaded_embeddings) if cache_data else None
        return np.array(final_embeddings)
