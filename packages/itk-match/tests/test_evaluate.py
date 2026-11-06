# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
import pytest
from itk_match.evaluate import (
    batch_groups,
    build_messages,
    evaluate_groups,
    parse_evaluations,
)

from tests.conftest import FakeChat


def _groups(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group_id": [i for i in range(n) for _ in range(2)],
            "group_size": [2] * (2 * n),
            "name_similarity": [1.0] * (2 * n),
            "record_id": [f"D{j}:{i}" for i in range(n) for j in (1, 2)],
            "name": ["Acme"] * (2 * n),
        },
        schema_overrides={"group_id": pl.UInt32},
    )


def test_build_messages_has_system_and_user_roles():
    messages = build_messages("group_id,name\n0,Acme")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "0,Acme" in messages[1]["content"]
    assert "JSON" in messages[0]["content"]


def test_parse_evaluations_accepts_fenced_json():
    text = '```json\n{"evaluations": [{"group_id": 3, "relatedness": 9}]}\n```'
    [evaluation] = parse_evaluations(text)
    assert evaluation.group_id == 3
    assert evaluation.relatedness == 9
    assert evaluation.explanation == ""


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"evaluations": [{"group_id": 1, "relatedness": 11}]}',
        '{"evaluations": [{"relatedness": 1}]}',
    ],
)
def test_parse_evaluations_rejects_bad_replies(text):
    with pytest.raises(ValueError, match="Could not parse"):
        parse_evaluations(text)


def test_batch_groups_keeps_whole_groups_together():
    batches = batch_groups(_groups(5), groups_per_batch=2)
    assert [b.get_column("group_id").unique().len() for b in batches] == [2, 2, 1]
    assert sum(b.height for b in batches) == 10


async def test_evaluate_groups_returns_one_row_per_group():
    chat = FakeChat()
    frame = await evaluate_groups(_groups(3), chat, groups_per_batch=2)

    assert frame.get_column("group_id").to_list() == [0, 1, 2]
    assert frame.get_column("relatedness").to_list() == [8, 8, 8]
    assert len(chat.messages) == 2


async def test_evaluate_groups_with_no_groups_makes_no_calls():
    chat = FakeChat()
    frame = await evaluate_groups(_groups(0), chat)
    assert frame.height == 0
    assert frame.columns == ["group_id", "relatedness", "explanation"]
    assert chat.messages == []
