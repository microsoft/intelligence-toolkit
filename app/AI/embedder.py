# Copyright (c) 2024 Microsoft Corporation. All rights reserved.

import logging
from typing import List

import numpy as np

from .cache_pickle import CachePickle
from .client import OpenAIClient
from .openai_configuration import OpenAIConfiguration
from .utils import get_token_count, hash_text

logger = logging.getLogger(__name__)

class Embedder:
    _openai_client = None
    pickle = None

    def __init__(self, configuration: OpenAIConfiguration = OpenAIConfiguration(), pickle_path = None) -> None:
        self.configuration = configuration
        self.openai_client = OpenAIClient(configuration)
        self.pickle = CachePickle(path=pickle_path)

    def embed_store_one(self, text: str, cache_data=True):
        hash = hash_text(text)
        loaded_embeddings = self.pickle.get_all() if cache_data else {}
        embedding = self.pickle.get(hash, loaded_embeddings)  if cache_data else {}

        if not embedding:
            tokens = get_token_count(text)
            if tokens > self.configuration.max_tokens:
                text = text[:self.configuration.max_tokens]
                logger.info('Truncated text to max tokens')
            try:
                embedding = self.openai_client.generate_embedding([text])
            except Exception as e:
                raise Exception(f'Problem in OpenAI response. {e}')
                
            loaded_embeddings.update({hash: embedding})
            self.pickle.save(loaded_embeddings) if cache_data else None
        return embedding
    
    def embed_store_many(self, texts: List[str], callback=None, cache_data=True):
        final_embeddings = [None] * len(texts)
        new_texts  = []
        loaded_embeddings = self.pickle.get_all() if cache_data else {}
        count = 0
        for ix, text in enumerate(texts):
            hash = hash_text(text)
            embedding = self.pickle.get(hash, loaded_embeddings) if cache_data else {}
            if not len(embedding):
                new_texts.append((ix, text))
            else:
                final_embeddings[ix] = np.array(embedding)
            count += 1
        print(f'Got {count} existing texts')
        logger.info(f'Got {count} existing texts')
        print(f'Got {len(new_texts)} new texts')
        logger.info(f'Got {len(new_texts)} new texts')

        num_batches = len(new_texts) // 2000 + 1
        batch_count = 1
        for i in range(0, len(new_texts), 2000):
            if callback:
                for cb in callback:
                    cb.on_embedding_batch_change(batch_count, num_batches)
            batch_count += 1
            batch = new_texts[i:i+2000]
            batch_texts = [x[1] for x in batch]
            try:
                embedding = [x.embedding for x in self.openai_client.generate_embeddings(batch_texts).data]
                loaded_embeddings.update({hash: embedding})
            except Exception as e:
                raise Exception(f'Problem in OpenAI response. {e}')
            
            for j, (ix, text) in enumerate(batch):
                hsh = hash_text(text)
                loaded_embeddings.update({hsh: embedding[j]})
                final_embeddings[ix] = np.array(embedding[j])
        self.pickle.save(loaded_embeddings) if cache_data else None
        return np.array(final_embeddings)