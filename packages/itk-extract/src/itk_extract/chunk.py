# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Cut sources into chunks small enough for one model call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from itk_extract.models import Chunk

if TYPE_CHECKING:
    from collections.abc import Callable

    from itk_extract.models import Source

    Splitter = Callable[[str], list[str]]
    """Turns one text into pieces; ``TextSplitter(...).split`` from itk-core fits."""


def token_splitter(chunk_tokens: int) -> Splitter:
    """The itk-core semantic splitter bound to a chunk size in tokens."""
    from itk_core.text_splitter import TextSplitter

    return TextSplitter(chunk_size=chunk_tokens).split


def chunk_sources(
    sources: list[Source], splitter: Splitter | None = None
) -> list[Chunk]:
    """One chunk per source, or several when a splitter is given.

    Chunk ids are ``<source_id>#<index>`` so they stay unique across sources.
    Blank pieces are dropped, so an empty source yields no chunks.
    """
    chunks: list[Chunk] = []
    for source in sources:
        pieces = splitter(source.text) if splitter else [source.text]
        for index, text in enumerate(p for p in pieces if p.strip()):
            chunks.append(
                Chunk(
                    id=f"{source.id}#{index}",
                    source_id=source.id,
                    index=index,
                    text=text,
                )
            )
    return chunks
