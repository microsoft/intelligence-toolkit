# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import pytest
from itk_match.embed import embed_sentences


async def test_embed_sentences_returns_aligned_float32_matrix(embedder):
    matrix = await embed_sentences(["NAME: A", "NAME: B", "NAME: A"], embedder)

    assert matrix.dtype == np.float32
    assert matrix.shape[0] == 3
    assert np.array_equal(matrix[0], matrix[2])
    assert not np.array_equal(matrix[0], matrix[1])


async def test_embed_sentences_embeds_duplicates_once(embedder):
    await embed_sentences(["x", "y", "x", "x"], embedder)
    assert embedder.calls == [["x", "y"]]


async def test_embed_sentences_handles_empty_input(embedder):
    matrix = await embed_sentences([], embedder)
    assert matrix.shape == (0, 0)
    assert embedder.calls == []


async def test_embed_sentences_fails_loudly_on_missing_vectors():
    class Partial:
        async def embed_store_many(self, data, callbacks=None, cache_data=True):
            return [{"text": data[0]["text"], "vector": [1.0]}]

    with pytest.raises(ValueError, match="no vector for 1 sentence"):
        await embed_sentences(["a", "b"], Partial())
