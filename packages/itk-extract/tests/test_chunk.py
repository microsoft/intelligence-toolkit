# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
from __future__ import annotations

from itk_extract.chunk import chunk_sources, token_splitter
from itk_extract.sources import from_texts


def test_no_splitter_gives_one_chunk_per_source() -> None:
    chunks = chunk_sources(from_texts(["one", "two"]))
    assert [(c.id, c.source_id, c.index, c.text) for c in chunks] == [
        ("text-0#0", "text-0", 0, "one"),
        ("text-1#0", "text-1", 0, "two"),
    ]


def test_splitter_fans_out_and_skips_blank_pieces() -> None:
    chunks = chunk_sources(from_texts(["a b c"]), splitter=lambda t: t.split(" "))
    assert [c.text for c in chunks] == ["a", "b", "c"]
    assert [c.id for c in chunks] == ["text-0#0", "text-0#1", "text-0#2"]
    assert chunk_sources(from_texts([""]), splitter=lambda t: [t, "  "]) == []


def test_empty_source_without_splitter_yields_nothing() -> None:
    assert chunk_sources(from_texts(["", "   "])) == []


def test_token_splitter_bounds_pieces() -> None:
    split = token_splitter(8)
    pieces = split("The quick brown fox jumps over the lazy dog. " * 10)
    assert len(pieces) > 1
    assert "".join(p.strip() for p in pieces).startswith("The quick brown fox")
