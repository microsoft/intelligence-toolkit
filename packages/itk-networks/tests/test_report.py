# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import polars as pl
from itk_networks import prompts
from itk_networks.report import (
    build_report_messages,
    generate_report,
    report_variables,
)
from itk_networks_testing import FakeChat

NODES = pl.DataFrame({"node": ["ENTITY==A"], "type": ["ENTITY"], "flags": [2]})
EDGES = pl.DataFrame({
    "source": ["ENTITY==A"],
    "target": ["phone==1"],
    "attribute": ["phone"],
})


def test_report_variables_carry_csv_frames():
    variables = report_variables(NODES, EDGES, 3, "A", "exposure md", 5, 2.5)
    assert variables["entity_id"] == "A"
    assert variables["network_id"] == 3
    assert variables["max_flags"] == 5
    assert variables["mean_flags"] == 2.5
    assert variables["exposure"] == "exposure md"
    assert variables["network_nodes"] == "node,type,flags\nENTITY==A,ENTITY,2\n"
    assert variables["network_edges"].startswith("source,target,attribute\n")


def test_messages_contain_prompt_variables_and_safety_text():
    messages = build_report_messages(report_variables(NODES, EDGES, 3, "A"))
    assert [m["role"] for m in messages] == ["system"]  # legacy: one system message
    text = " ".join(m["content"] for m in messages)
    assert "Selected entity: A" in text
    assert "Selected network: 3" in text
    assert "ENTITY==A,ENTITY,2" in text
    assert prompts.USER_PROMPT.splitlines()[0] in text
    assert prompts.SAFETY_PROMPT in text


def test_custom_instructions_replace_user_prompt():
    messages = build_report_messages(report_variables(NODES, EDGES, 0), "Be brief.")
    text = " ".join(m["content"] for m in messages)
    assert "Be brief." in text
    assert "Goal:" not in text


async def test_generate_report_calls_chat_without_streaming():
    chat = FakeChat("report text")
    messages = build_report_messages(report_variables(NODES, EDGES, 0))
    assert await generate_report(messages, chat) == "report text"
    assert chat.messages == [messages]
    assert chat.kwargs == [{"stream": False, "callbacks": None}]
