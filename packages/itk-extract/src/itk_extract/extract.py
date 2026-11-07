# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Send chunks to the model and turn replies into validated objects."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from itk_extract.models import Extraction, ExtractionError
from itk_extract.prompt import build_messages

if TYPE_CHECKING:
    from itk_core.progress import ProgressBatchCallback

    from itk_extract.models import Chunk, ExtractConfig
    from itk_extract.schema import RecordSchema

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_MAX_REPORTED_VIOLATIONS = 5


class BatchChatFn(Protocol):
    """Matches ``itk_core.base_chat.BaseChat.generate_texts_async``."""

    async def __call__(
        self,
        messages_list: list[list[dict[str, str]]],
        callbacks: list[ProgressBatchCallback] | None = ...,
        **llm_kwargs: Any,
    ) -> list[str]: ...


def parse_reply(
    text: str, schema: RecordSchema, validate: bool = True
) -> dict[str, Any]:
    """Parse one model reply into an object, tolerating a code fence around it.

    Raises ``ValueError`` when the reply is not JSON or (when ``validate`` is
    set) violates the schema, and ``TypeError`` when it is not a JSON object.
    """
    cleaned = _FENCE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        msg = f"Reply is not valid JSON: {e.msg} at position {e.pos}"
        raise ValueError(msg) from e
    if not isinstance(data, dict):
        msg = f"Reply is a JSON {type(data).__name__}, not an object"
        raise TypeError(msg)
    if validate:
        violations = schema.validate(data)
        if violations:
            shown = "; ".join(violations[:_MAX_REPORTED_VIOLATIONS])
            extra = len(violations) - _MAX_REPORTED_VIOLATIONS
            if extra > 0:
                shown += f"; and {extra} more"
            msg = f"Reply violates schema: {shown}"
            raise ValueError(msg)
    return data


async def extract_chunks(
    chunks: list[Chunk],
    chat: BatchChatFn,
    schema: RecordSchema,
    config: ExtractConfig,
    callbacks: list[ProgressBatchCallback] | None = None,
) -> tuple[list[Extraction], list[ExtractionError]]:
    """One model call per distinct chunk text; replies parsed per chunk.

    Unusable replies become ``ExtractionError`` entries unless
    ``config.fail_fast`` is set, in which case the first one raises.
    """
    if not chunks:
        return [], []
    unique_texts = list(dict.fromkeys(chunk.text for chunk in chunks))
    messages_list = [build_messages(text, config.guidance) for text in unique_texts]
    replies = await chat(
        messages_list, callbacks=callbacks, response_format=schema.response_format()
    )
    if len(replies) != len(unique_texts):
        msg = f"Chat returned {len(replies)} replies for {len(unique_texts)} prompts"
        raise ValueError(msg)
    reply_by_text = dict(zip(unique_texts, replies, strict=True))

    extractions: list[Extraction] = []
    errors: list[ExtractionError] = []
    for chunk in chunks:
        raw = reply_by_text[chunk.text]
        raw_text = raw if isinstance(raw, str) else json.dumps(raw)
        try:
            data = parse_reply(raw_text, schema, config.validate_output)
        except (ValueError, TypeError) as e:
            if config.fail_fast:
                msg = f"Chunk {chunk.id}: {e}"
                raise ValueError(msg) from e
            errors.append(ExtractionError(chunk, str(e), raw_text))
            continue
        extractions.append(Extraction(chunk, data))
    return extractions, errors
