# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Messages sent to the model for one chunk."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You extract a structured JSON object from unstructured text, following the \
JSON schema you are given.

Include all information from the input text that the schema has a place for. \
Do not fabricate information that is not present in the input text. Leave \
fields empty or null when the text does not provide them.
{guidance_section}"""

GUIDANCE_SECTION = """
--Extraction guidance--

{guidance}
"""

USER_PROMPT = """\
--Input text--

{text}
"""


def build_messages(text: str, guidance: str = "") -> list[dict[str, str]]:
    """System message with the instructions, user message with the text."""
    guidance = guidance.strip()
    section = GUIDANCE_SECTION.format(guidance=guidance) if guidance else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(guidance_section=section)},
        {"role": "user", "content": USER_PROMPT.format(text=text)},
    ]
