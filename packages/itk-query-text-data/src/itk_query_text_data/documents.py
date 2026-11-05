# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Load files into documents and split documents into a chunked corpus."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from itk_core.defaults import CHUNK_SIZE
from itk_core.text_splitter import TextSplitter

from itk_query_text_data.models import Chunk, Corpus, Document, Progress

EventHandler = Callable[[Progress], None]


def source_name(path: str | Path) -> str:
    """Display name for a file: its base name with brackets removed and spaces as underscores."""
    name = re.split(r"[\\/]", str(path))[-1]
    return name.replace("(", "").replace(")", "").replace(" ", "_")


def read_documents(
    paths: Iterable[str | Path],
    *,
    on_event: EventHandler | None = None,
) -> list[Document]:
    """Read text, CSV, JSON and PDF files.

    Each CSV row and each item of a top-level JSON list becomes its own document
    named ``"<file>_<n>"``.
    """
    paths = list(paths)
    documents: list[Document] = []
    for position, path in enumerate(paths, start=1):
        if on_event is not None:
            on_event(Progress("read", position, len(paths)))
        documents.extend(_read_file(Path(path)))
    return documents


def _read_file(path: Path) -> list[Document]:
    name = source_name(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        return [
            Document(
                f"{name}_{row_number}",
                "; ".join(f"{column}: {row[column]}" for column in frame.columns),
            )
            for row_number, (_, row) in enumerate(frame.iterrows(), start=1)
        ]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [
                Document(f"{name}_{item_number}", json.dumps(item))
                for item_number, item in enumerate(data, start=1)
            ]
        return [Document(name, json.dumps(data))]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return [Document(name, " ".join(page.extract_text() for page in reader.pages))]
    return [Document(name, path.read_text(encoding="utf-8"))]


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = CHUNK_SIZE
) -> Corpus:
    """Split documents into token-bounded chunks.

    Documents that share a source name are treated as one source, so chunk
    indices keep counting across them.
    """
    splitter = TextSplitter(chunk_size=chunk_size)
    by_source: dict[str, list[tuple[str, str | None]]] = {}
    for document in documents:
        pieces = by_source.setdefault(document.source, [])
        pieces.extend((text, document.period) for text in splitter.split(document.text))
    chunks: list[Chunk] = []
    for source, pieces in by_source.items():
        for index, (text, period) in enumerate(pieces, start=1):
            chunks.append(Chunk(len(chunks) + 1, source, index, text, period))
    return Corpus(tuple(chunks))


def load_documents(
    paths: Iterable[str | Path],
    chunk_size: int = CHUNK_SIZE,
    *,
    on_event: EventHandler | None = None,
) -> Corpus:
    """Read files and chunk them in one step."""
    return chunk_documents(read_documents(paths, on_event=on_event), chunk_size)
