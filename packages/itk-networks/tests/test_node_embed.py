# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import numpy as np
import pytest
from itk_networks.embed import embed_nodes, nodes_to_embed
from itk_networks.prepare import attribute_links, build_graph
from itk_networks_testing import A, E, FakeEmbedder, attribute_table, letter_bag


@pytest.fixture()
def graph():
    return build_graph(attribute_links(attribute_table()))


def test_nodes_to_embed_filters_by_type_and_sorts(graph):
    pairs = nodes_to_embed(graph, ["ENTITY"])
    assert pairs == [(E(n), "ENTITY") for n in "ABCDEF"]
    both = nodes_to_embed(graph, ["ENTITY", "phone"])
    assert both[0] == (E("A"), "ENTITY")
    assert both[-1] == (A("phone", "444"), "phone")


def test_nodes_to_embed_requires_types(graph):
    with pytest.raises(ValueError, match="No node types"):
        nodes_to_embed(graph, [])


async def test_embed_nodes_returns_aligned_matrix(graph):
    embedder = FakeEmbedder()
    texts, matrix = await embed_nodes(graph, ["ENTITY"], embedder)
    assert texts == [E(n) for n in "ABCDEF"]
    assert matrix.shape == (6, 36)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[2], np.asarray(letter_bag(E("C")), np.float32))
    assert embedder.calls == [texts]


async def test_embed_nodes_passes_type_in_additional_details(graph):
    class Recording(FakeEmbedder):
        async def embed_store_many(self, data, callbacks=None, cache_data=True):
            self.payload = data
            return await super().embed_store_many(data, callbacks, cache_data)

    embedder = Recording()
    await embed_nodes(graph, ["phone"], embedder)
    assert {d["additional_details"]["type"] for d in embedder.payload} == {"phone"}
    assert all(d["hash"] for d in embedder.payload)


async def test_embed_nodes_with_unknown_type_is_empty(graph):
    texts, matrix = await embed_nodes(graph, ["zip"], FakeEmbedder())
    assert texts == []
    assert matrix.shape == (0, 0)


async def test_embed_nodes_rejects_missing_vectors(graph):
    class Dropping(FakeEmbedder):
        async def embed_store_many(self, data, callbacks=None, cache_data=True):
            return (await super().embed_store_many(data, callbacks, cache_data))[1:]

    with pytest.raises(ValueError, match="no vector for 1 node"):
        await embed_nodes(graph, ["ENTITY"], Dropping())
