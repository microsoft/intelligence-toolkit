# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from python.helpers.constants import (
    VECTOR_STORE_MAX_RETRIES,
    VECTOR_STORE_MAX_RETRIES_WAIT_TIME,
)

T = TypeVar("T")


def retry_with_backoff(
    _func=None,
    *,
    retries=VECTOR_STORE_MAX_RETRIES,
    backoff_in_seconds=VECTOR_STORE_MAX_RETRIES_WAIT_TIME,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise
                    sleep = backoff_in_seconds * 2**x + random.uniform(0, 1)
                    time.sleep(sleep)
                    x += 1

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)
