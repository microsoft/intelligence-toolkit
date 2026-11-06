# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pytest
from itk_match import MatchConfig, Matcher, MatchResult


async def test_match_groups_identical_records_across_datasets(
    embedder, dataset_one, dataset_two
):
    matcher = Matcher(embedder)
    result = await matcher.match(
        [dataset_one, dataset_two], MatchConfig(max_distance=0.2)
    )

    frame = result.groups_frame
    adventure = frame.filter(frame["name"] == "Adventure Gear")
    assert adventure.get_column("record_id").to_list() == ["D1:1", "D2:0"]
    assert adventure.get_column("group_id").n_unique() == 1
    assert result.embeddings is not None
    assert result.embeddings.shape[0] == result.records.height


async def test_regroup_tightens_without_re_embedding(
    embedder, dataset_one, dataset_two
):
    matcher = Matcher(embedder)
    loose = await matcher.match(
        [dataset_one, dataset_two], MatchConfig(max_distance=0.2)
    )
    calls_after_match = len(embedder.calls)

    tight = Matcher.regroup(
        loose, MatchConfig(max_distance=0.2, min_name_similarity=0.9)
    )

    assert len(embedder.calls) == calls_after_match
    assert len(tight.groups) <= len(loose.groups)
    assert all(g.name_similarity >= 0.9 for g in tight.groups)
    assert tight.config.min_name_similarity == 0.9
    # Sprout Ltd / Spout Ltd (trigram 0.5) are grouped loosely but not tightly.
    loose_names = set(loose.groups_frame.get_column("name").to_list())
    tight_names = set(tight.groups_frame.get_column("name").to_list())
    assert {"Sprout Ltd", "Spout Ltd"} <= loose_names
    assert not ({"Sprout Ltd", "Spout Ltd"} & tight_names)


def test_regroup_requires_embeddings(dataset_one):
    from itk_match.unify import unify

    bare = MatchResult(unify([dataset_one]), [], [], MatchConfig())
    with pytest.raises(ValueError, match="no embeddings"):
        Matcher.regroup(bare, MatchConfig())


async def test_evaluate_and_integrate(embedder, fake_chat, dataset_one, dataset_two):
    chat = fake_chat
    matcher = Matcher(embedder, chat)
    result = await matcher.match(
        [dataset_one, dataset_two], MatchConfig(max_distance=0.2)
    )

    evaluations = await matcher.evaluate(result)
    integrated = Matcher.integrate(result, evaluations)

    assert evaluations.height == len(result.groups)
    assert integrated.height == result.groups_frame.height
    assert integrated.get_column("relatedness").null_count() == 0
    assert "explanation" in integrated.columns


async def test_evaluate_without_chat_is_an_error(embedder, dataset_one):
    matcher = Matcher(embedder)
    result = await matcher.match([dataset_one])
    with pytest.raises(ValueError, match="no chat function"):
        await matcher.evaluate(result)


async def test_pairs_frame_mirrors_pairs(embedder, dataset_one, dataset_two):
    result = await Matcher(embedder).match(
        [dataset_one, dataset_two], MatchConfig(max_distance=0.2)
    )
    frame = result.pairs_frame
    assert frame.height == len(result.pairs)
    assert frame.columns == ["left", "right", "distance", "name_similarity"]
