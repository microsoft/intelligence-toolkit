# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio

import polars as pl
from itk_patterns import prompts
from itk_patterns.report import (
    build_report_messages,
    generate_report,
    report_variables,
)
from itk_patterns_testing import FakeChat


def test_report_variables_render_frames_as_csv():
    series = pl.DataFrame({"period": ["P1"], "pattern": ["a=1"], "count": [2]})
    counts = pl.DataFrame({"attribute_value": ["a=1"], "count": [2]})
    variables = report_variables("a=1", "P1", series, counts)
    assert variables["pattern"] == "a=1"
    assert variables["period"] == "P1"
    assert variables["time_series"].splitlines() == ["period,pattern,count", "P1,a=1,2"]
    assert variables["attribute_counts"].splitlines() == [
        "attribute_value,count",
        "a=1,2",
    ]


def test_build_report_messages_fills_prompt_and_appends_safety():
    messages = build_report_messages({
        "pattern": "a=1 & b=2",
        "period": "P9",
        "time_series": "ts",
        "attribute_counts": "ac",
    })
    # generate_messages folds everything into one system message
    assert [m["role"] for m in messages] == ["system"]
    system = messages[0]["content"]
    assert "Detected pattern: a=1 & b=2" in system
    assert "Detected in period: P9" in system
    assert "# Pattern Report" in system
    assert prompts.SAFETY_PROMPT.strip() in system


def test_build_report_messages_takes_custom_instructions():
    messages = build_report_messages(
        {"pattern": "p", "period": "q", "time_series": "", "attribute_counts": ""},
        instructions="Reply in one line.",
    )
    assert "Reply in one line." in messages[0]["content"]
    assert "# Pattern Report" not in messages[0]["content"]


def test_generate_report_calls_chat_without_streaming():
    chat = FakeChat("done")
    messages = [{"role": "system", "content": "x"}]
    assert asyncio.run(generate_report(messages, chat)) == "done"
    assert chat.messages == [messages]
