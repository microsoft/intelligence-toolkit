# itk-query-text-data

Query Text Data (v2) for Intelligence Toolkit: concept graphs, relevance search and grounded answers over text corpora.

Each step is a standalone function that takes and returns typed records, with no Streamlit or session state:

```python
import itk_query_text_data as qtd

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

It is not published or bundled in the `intelligence-toolkit` wheel yet; `uv sync` installs it into the workspace through the root `dev` dependency group.

Tests: `uv run pytest packages/itk-query-text-data/tests` (set `UPDATE_GOLDEN=1` to regenerate `tests/golden`).
