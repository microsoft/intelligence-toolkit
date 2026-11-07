# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from itk_extract.prompt import build_messages


def test_text_goes_in_user_message() -> None:
    messages = build_messages("Alice runs Acme.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Alice runs Acme." in messages[1]["content"]
    assert "Alice runs Acme." not in messages[0]["content"]
    assert "Do not fabricate" in messages[0]["content"]


def test_guidance_is_optional() -> None:
    without = build_messages("x")[0]["content"]
    with_guidance = build_messages("x", guidance="Use ISO dates.")[0]["content"]
    assert "Extraction guidance" not in without
    assert "Extraction guidance" in with_guidance
    assert "Use ISO dates." in with_guidance
    assert build_messages("x", guidance="   ")[0]["content"] == without


def test_braces_in_text_survive_formatting() -> None:
    messages = build_messages('{"a": 1}', guidance="{keep}")
    assert '{"a": 1}' in messages[1]["content"]
    assert "{keep}" in messages[0]["content"]
