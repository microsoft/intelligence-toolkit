# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Token usage reporting for model calls made through `OpenAIClient`.

Workflows create their own clients, so callbacks are registered for the whole
process instead of being passed to each client:

    from itk_core.usage import LLMUsage, usage_callback

    calls: list[LLMUsage] = []
    with usage_callback(calls.append):
        ...  # run any workflow
    total = sum(call.total_tokens or 0 for call in calls)

Cost is left to the caller, who knows the prices that apply to their deployment.
"""

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMUsage:
    """Token usage of one successful model call.

    A count is None when the provider did not report it; it is never guessed.
    """

    operation: Literal["chat", "embedding"]
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


UsageCallback = Callable[[LLMUsage], None]

_callbacks: list[UsageCallback] = []


def add_usage_callback(callback: UsageCallback) -> None:
    """Call `callback` after every successful model call in this process."""
    _callbacks.append(callback)


def remove_usage_callback(callback: UsageCallback) -> None:
    """Stop calling `callback`. Does nothing if it is not registered."""
    with suppress(ValueError):
        _callbacks.remove(callback)


@contextmanager
def usage_callback(callback: UsageCallback) -> Generator[None, None, None]:
    """Register `callback` for the duration of a `with` block."""
    add_usage_callback(callback)
    try:
        yield
    finally:
        remove_usage_callback(callback)


def has_usage_callbacks() -> bool:
    return bool(_callbacks)


def _count(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def report_usage(
    operation: Literal["chat", "embedding"], model: str, usage: Any
) -> None:
    """Send the SDK `usage` object of a finished call to every callback.

    A callback that raises is logged and skipped, so reporting never fails the
    model call it describes.
    """
    if not _callbacks:
        return
    record = LLMUsage(
        operation=operation,
        model=model,
        input_tokens=_count(usage, "prompt_tokens"),
        output_tokens=_count(usage, "completion_tokens"),
        total_tokens=_count(usage, "total_tokens"),
    )
    for callback in tuple(_callbacks):
        try:
            callback(record)
        except Exception:
            log.exception("Usage callback failed")
