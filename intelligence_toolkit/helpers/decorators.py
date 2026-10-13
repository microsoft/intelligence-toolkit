# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio
import inspect
import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from intelligence_toolkit.helpers.constants import (
    VECTOR_STORE_MAX_RETRIES,
    VECTOR_STORE_MAX_RETRIES_WAIT_TIME,
)

T = TypeVar("T")

log = logging.getLogger(__name__)


def _compute_delay(
    attempt: int,
    backoff_in_seconds: float,
    error: Exception,
    delay_hint: Callable[[Exception], float | None] | None,
) -> float:
    if delay_hint is not None:
        hinted = delay_hint(error)
        if hinted is not None:
            return hinted
    return backoff_in_seconds * 2**attempt + random.uniform(0, 1)


def retry_with_backoff(
    retries=VECTOR_STORE_MAX_RETRIES,
    backoff_in_seconds=VECTOR_STORE_MAX_RETRIES_WAIT_TIME,
    retry_on: Callable[[Exception], bool] | None = None,
    delay_hint: Callable[[Exception], float | None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a sync or async function with exponential backoff.

    Args:
        retries: Number of retries after the first attempt.
        backoff_in_seconds: Base delay; doubles on each retry (plus jitter).
        retry_on: Predicate deciding whether an exception is retryable.
            Defaults to retrying every exception.
        delay_hint: Optional function returning a server-suggested delay
            (e.g. Retry-After) that overrides the computed backoff.
    """

    def should_retry(error: Exception, attempt: int) -> bool:
        if attempt >= retries:
            return False
        return retry_on is None or retry_on(error)

    def decorator(func) -> Any:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                attempt = 0
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if not should_retry(e, attempt):
                            raise
                        delay = _compute_delay(
                            attempt, backoff_in_seconds, e, delay_hint
                        )
                        log.warning(
                            "%s failed (attempt %s/%s), retrying in %.1fs: %s",
                            func.__qualname__,
                            attempt + 1,
                            retries + 1,
                            delay,
                            e,
                        )
                        await asyncio.sleep(delay)
                        attempt += 1

            return async_wrapper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not should_retry(e, attempt):
                        raise
                    delay = _compute_delay(attempt, backoff_in_seconds, e, delay_hint)
                    log.warning(
                        "%s failed (attempt %s/%s), retrying in %.1fs: %s",
                        getattr(func, "__qualname__", repr(func)),
                        attempt + 1,
                        retries + 1,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                    attempt += 1

        return wrapper

    return decorator
