# itk-query

Query Text Data (v2) for Intelligence Toolkit: concept graphs, relevance search and grounded answers over text corpora.

Each step is a standalone function that takes and returns typed records, with no Streamlit or session state. For a runnable end-to-end walkthrough, see [example_notebooks/query_text_data_v2.ipynb](../../example_notebooks/query_text_data_v2.ipynb).

```python
import itk_query as qtd

llm = qtd.OpenAILLM(openai_configuration)   # any object matching qtd.LLM
embedder = qtd.ItkEmbedder(itk_embedder)     # any object matching qtd.Embedder

corpus = qtd.load_documents(["report.pdf", "notes.txt"], chunk_size=500)
graph = qtd.build_concept_graph(corpus)
embeddings = await qtd.embed_chunks(corpus, embedder)

result = await qtd.search(query, corpus, graph, embeddings, llm=llm, embedder=embedder)
answer = await qtd.build_answer(result, corpus, llm)
print(answer.markdown)
```

`OpenAILLM` and `ItkEmbedder` adapt `itk_core` clients; tests use fakes. Pass `on_event` to receive `Progress`, `ChunksTested` and `AnalysisUpdated` events.

## Differences from v1 (`intelligence_toolkit.query_text_data`)

Output matches v1 on the characterization corpus (see `tests/test_qtd2_parity.py`), except:

- Adjacent-chunk search only follows neighbours within the same source file.
- Citations to ids that are not in the corpus are dropped instead of rendered as broken links.
- Concept communities are deterministic regardless of `PYTHONHASHSEED`.
- Importing does not call `nltk.download`.
- Empty corpora and corpora with no concepts return empty results instead of crashing.
- With live thematic analysis on, relevant chunks the model leaves out of every theme are kept in a final `"Other relevant chunks"` group (`"All relevant chunks"` when there are no themes) instead of being dropped from the report.

The itk_core adapters are also hardened: `ItkEmbedder` aligns vectors to texts by content hash (not position) and fails loudly when one is missing, `OpenAILLM` strips the streaming cursor glyph before calling `on_token`, and malformed thematic-update or report-integration replies are logged and tolerated rather than raised.

Like `itk-core` and `itk-match` it is not published on its own; it ships inside the `intelligence-toolkit` wheel and is importable from any workspace checkout.

Tests: `uv run pytest packages/itk-query/tests` (set `UPDATE_GOLDEN=1` to regenerate `tests/golden`).
