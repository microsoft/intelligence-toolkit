# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Error types and transient-error detection for the AI layer."""

import asyncio

import openai

from itk_core.defaults import LLM_MAX_RETRIES, LLM_RETRY_BASE_SECONDS
from itk_core.retry import retry_with_backoff

MAX_RETRY_AFTER_SECONDS = 60.0

_TRANSIENT_OPENAI_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class LLMError(Exception):
    """Raised when a chat completion request fails."""


class EmbeddingError(Exception):
    """Raised when an embedding request fails."""


def _causes(error: BaseException):
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_error(error: BaseException) -> bool:
    """Return True if the error (or anything in its cause chain) is worth retrying."""
    for err in _causes(error):
        if isinstance(err, _TRANSIENT_OPENAI_ERRORS):
            return True
        if isinstance(err, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
        if isinstance(err, openai.APIStatusError):
            status = err.status_code
            if status in (408, 409, 429) or status >= 500:
                return True
    return False


def retry_after_seconds(error: BaseException) -> float | None:
    """Extract a server-provided Retry-After delay from an OpenAI error, if any."""
    for err in _causes(error):
        response = getattr(err, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            continue
        for header in ("retry-after-ms", "retry-after"):
            value = headers.get(header)
            if value is None:
                continue
            try:
                seconds = float(value) / (1000 if header.endswith("-ms") else 1)
            except (TypeError, ValueError):
                continue
            if seconds >= 0:
                return min(seconds, MAX_RETRY_AFTER_SECONDS)
    return None


def retry_transient(retries: int = LLM_MAX_RETRIES, backoff_in_seconds: float = LLM_RETRY_BASE_SECONDS):
    """Retry decorator (sync or async) that only retries transient OpenAI failures."""
    return retry_with_backoff(
        retries=retries,
        backoff_in_seconds=backoff_in_seconds,
        retry_on=is_transient_error,
        delay_hint=retry_after_seconds,
    )
