# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
"""Query Text Data (v2): answer questions over a text corpus with grounded, cited reports.

Each step is a plain function over immutable records::

    corpus = (
        load_documents(
            paths
        )
    )
    graph = build_concept_graph(
        corpus
    )
    embeddings = await embed_chunks(
        corpus, embedder
    )
    expanded = await rewrite_query(
        query,
        graph,
        llm,
    )
    result = await search(
        query,
        corpus,
        graph,
        embeddings,
        llm=llm,
        embedder=embedder,
        expanded_query=expanded,
    )
    answer = await build_answer(
        result,
        corpus,
        llm,
    )
"""

from itk_query_text_data.analysis import generate_commentary
from itk_query_text_data.answer import build_answer, condense_answer, link_sources
from itk_query_text_data.concepts import build_concept_graph, extract_concepts
from itk_query_text_data.documents import (
    chunk_documents,
    load_documents,
    read_documents,
    source_name,
)
from itk_query_text_data.llm import LLM, Embedder, ItkEmbedder, OpenAILLM
from itk_query_text_data.models import (
    AnalysisUpdated,
    Answer,
    Chunk,
    ChunksTested,
    Community,
    ConceptGraph,
    Corpus,
    Document,
    Event,
    Progress,
    RelevanceTest,
    SearchConfig,
    SearchResult,
    ThemeStructure,
)
from itk_query_text_data.search import (
    embed_chunks,
    format_search_summary,
    rank_chunks,
    rewrite_query,
    search,
)

__all__ = [
    "LLM",
    "AnalysisUpdated",
    "Answer",
    "Chunk",
    "ChunksTested",
    "Community",
    "ConceptGraph",
    "Corpus",
    "Document",
    "Embedder",
    "Event",
    "ItkEmbedder",
    "OpenAILLM",
    "Progress",
    "RelevanceTest",
    "SearchConfig",
    "SearchResult",
    "ThemeStructure",
    "build_answer",
    "build_concept_graph",
    "chunk_documents",
    "condense_answer",
    "embed_chunks",
    "extract_concepts",
    "format_search_summary",
    "generate_commentary",
    "link_sources",
    "load_documents",
    "rank_chunks",
    "read_documents",
    "rewrite_query",
    "search",
    "source_name",
]
