# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Turn sentences into an aligned embedding matrix through an itk-core embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from itk_core.utils import hash_text

if TYPE_CHECKING:
    from itk_core.classes import VectorData
    from itk_core.progress import ProgressBatchCallback


class Embedder(Protocol):
    """The slice of itk_core.base_embedder.BaseEmbedder this package relies on."""

    async def embed_store_many(
        self,
        data: list[VectorData],
        callbacks: list[ProgressBatchCallback] | None = None,
        cache_data: bool = True,
    ) -> Any: ...


async def embed_sentences(
    sentences: list[str],
    embedder: Embedder,
    callbacks: list[ProgressBatchCallback] | None = None,
) -> np.ndarray:
    """Return a float32 matrix with one row per sentence, in input order.

    Duplicate sentences are embedded once and the vector reused.
    """
    if not sentences:
        return np.zeros((0, 0), dtype=np.float32)

    unique = list(dict.fromkeys(sentences))
    payload: list[Any] = [{"text": text, "hash": hash_text(text)} for text in unique]
    stored = await embedder.embed_store_many(payload, callbacks=callbacks)
    by_text = {
        item["text"]: np.asarray(item["vector"], dtype=np.float32) for item in stored
    }
    missing = [t for t in unique if t not in by_text]
    if missing:
        msg = f"Embedder returned no vector for {len(missing)} sentence(s)"
        raise ValueError(msg)
    return np.vstack([by_text[text] for text in sentences])
