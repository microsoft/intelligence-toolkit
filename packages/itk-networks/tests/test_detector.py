# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_networks import NetworkConfig, NetworkDetector
from itk_networks.models import ModelSummary, Network
from itk_networks_testing import (
    A,
    E,
    FakeEmbedder,
    attribute_table,
)


def test_prepare_builds_graph_flags_and_groups(prepared):
    assert prepared.summary == ModelSummary(
        entities=6, attributes=9, flags=4, groups=2, links=17
    )
    assert str(prepared.summary).startswith("Number of entities: 6")
    assert prepared.max_entity_flags == 2
    assert prepared.mean_flagged_flags == 1.33
    assert prepared.entity_types == ["ENTITY", "city", "email", "phone"]
    assert prepared.networks == []
    assert prepared.inferred_links == {}


def test_prepare_with_attributes_only():
    result = NetworkDetector.prepare([attribute_table()])
    assert result.flags.is_empty()
    assert result.group_links.is_empty()
    assert result.summary.flags == 0


def test_prepare_concatenates_several_tables():
    extra = pl.DataFrame({
        "name": ["G"],
        "phone": ["111"],
        "email": [None],
        "city": ["Metro"],
    })
    result = NetworkDetector.prepare([attribute_table(), attribute_table(extra)])
    assert result.graph.has_edge(E("G"), A("phone", "111"))


async def test_index_then_infer_without_re_embedding(detector, embedder, prepared):
    indexed = await detector.index(prepared, ["ENTITY"])
    assert indexed.indexed_texts == tuple(E(n) for n in "ABCDEF")
    assert indexed.embeddings.shape == (6, 36)
    assert indexed.neighbor_indices.shape == (6, 6)  # k capped at six nodes
    assert embedder.calls == [list(indexed.indexed_texts)]

    loose = detector.infer(indexed, 1.0)
    tight = detector.infer(loose, 0.001)
    assert len(loose.inferred_links) == 6
    assert tight.inferred_links == {}
    assert tight.similarity_threshold == 0.001
    assert embedder.calls == [list(indexed.indexed_texts)]
    assert prepared.inferred_links == {}  # inputs are never mutated


async def test_infer_links_frame_strips_prefix(detector, prepared):
    indexed = await detector.index(prepared, ["ENTITY"])
    frame = detector.infer(indexed, 1.0).inferred_links_frame
    assert frame.columns == ["text", "similar"]
    assert frame.row(0) == ("A", "B")


def test_infer_requires_index(prepared):
    with pytest.raises(ValueError, match="not been indexed"):
        NetworkDetector.infer(prepared, 0.5)


async def test_index_requires_embedder(prepared):
    with pytest.raises(ValueError, match="no embedder"):
        await NetworkDetector().index(prepared, ["ENTITY"])


async def test_clear_inferred(detector, prepared):
    indexed = await detector.index(prepared, ["ENTITY"])
    cleared = detector.clear_inferred(detector.infer(indexed, 1.0))
    assert cleared.inferred_links == {}
    assert cleared.embeddings is not None


def test_identify_networks_and_records(identified):
    assert identified.networks == [
        Network(0, (E("A"), E("B"))),
        Network(1, (E("C"), E("D"))),
    ]
    assert identified.trimmed_attributes.rows() == [(A("city", "Metro"), 6)]
    assert identified.trimmed_nodes == {A("city", "Metro")}
    assert identified.records_summary == (
        "Networks identified: 2 (2 with multiple entities, maximum 2)"
    )
    frame = identified.entities_frame
    assert frame.get_column("entity_id").to_list() == ["A", "B", "C", "D"]
    assert frame.get_column("network_flags").to_list() == [2, 2, 1, 1]
    grouped = identified.grouped_frame
    assert grouped.get_column("sector").to_list() == [
        "retail",
        "retail",
        "mining",
        "mining",
    ]


def test_identify_default_config_links_everyone_through_city(detector, prepared):
    result = detector.identify(prepared)
    assert result.config == NetworkConfig()
    assert len(result.networks) == 1
    assert result.networks[0].size == 6


def test_identify_uses_detector_for_oversized_components(detector, chunker, prepared):
    result = detector.identify(prepared, NetworkConfig(max_network_entities=4))
    assert chunker.calls == [(6, 4, 42)]
    assert [n.entities for n in result.networks] == [
        (E("A"), E("B"), E("C"), E("D")),
        (E("E"), E("F")),
    ]


def test_identify_can_be_rerun_with_new_config(detector, identified):
    again = detector.identify(
        identified,
        NetworkConfig(max_attribute_degree=3, removed_attributes=(A("phone", "111"),)),
    )
    assert again.networks == [Network(0, (E("C"), E("D")))]
    assert identified.networks[0].entities == (E("A"), E("B"))


async def test_identify_merges_inferred_links(detector, prepared, config):
    unit = [[float(i == j) for j in range(6)] for i in range(6)]
    vectors = {E(n): unit[i] for i, n in enumerate("ABCD")}
    vectors[E("E")] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    vectors[E("F")] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0001]
    embedder = FakeEmbedder(vectors)
    local = NetworkDetector(embedder, detector.community_detector)
    indexed = await local.index(prepared, ["ENTITY"])
    inferred = local.infer(indexed, 0.001)
    assert inferred.inferred_links[E("E")] == frozenset({E("F")})
    result = local.identify(inferred, config)
    assert Network(2, (E("E"), E("F"))) in result.networks
    assert result.entities_frame.filter(pl.col("entity_id") == "F").row(0)[2] == 2


def test_explore_and_exposure(identified):
    sub = NetworkDetector.explore(identified, 0)
    assert set(sub.nodes()) == {
        E("A"),
        E("B"),
        A("phone", "111"),
        A("email", "a@x"),
        A("email", "b@x"),
    }
    exposure = NetworkDetector.exposure(identified, 0, "B")
    assert exposure.summary.direct == 0
    assert exposure.summary.indirect == 2
    assert exposure.summary.paths == 1
    assert exposure.paths[0].steps == ((E("A"),), (A("phone", "111"),), (E("B"),))
    assert "**B** has **0** direct flags" in exposure.markdown


def test_explore_excludes_removed_attributes(detector, prepared):
    config = NetworkConfig(
        max_attribute_degree=3, removed_attributes=(A("email", "a@x"),)
    )
    result = detector.identify(prepared, config)
    assert A("email", "a@x") not in NetworkDetector.explore(result, 0)


def test_report_messages_include_simplified_graph_and_exposure(identified):
    messages = NetworkDetector.report_messages(identified, 0, "B")
    text = " ".join(m["content"] for m in messages)
    assert "Selected entity: B" in text
    assert f"{E('A')},ENTITY,2" in text
    assert "email==a@x" not in text  # single-degree attribute dropped by simplify
    assert "Flag Exposure Paths" in text
    assert "Maximum flags of a flagged entity: 2" in text


async def test_report_uses_chat(detector, fake_chat, identified):
    report = await detector.report(identified, 0)
    assert report == fake_chat.reply
    text = " ".join(m["content"] for m in fake_chat.messages[0])
    assert "Selected entity: \n" in text
    assert "Flag exposure:\n\n\n" in text


async def test_report_requires_chat(identified):
    with pytest.raises(ValueError, match="no chat function"):
        await NetworkDetector().report(identified, 0)


def test_unknown_network_id(identified):
    with pytest.raises(ValueError, match="No network 9"):
        NetworkDetector.explore(identified, 9)
