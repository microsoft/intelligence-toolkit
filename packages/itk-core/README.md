# itk-core

Shared AI core for Intelligence Toolkit workflows:

- `itk_core.client` / `itk_core.base_chat` / `itk_core.utils`: OpenAI and Azure OpenAI chat, with transient-only retry, streaming and bounded concurrency.
- `itk_core.openai_embedder` / `itk_core.local_embedder`: embedders backed by a LanceDB `VectorStore` cache.
- `itk_core.errors`: typed `LLMError` and `EmbeddingError`.

It has no Streamlit, torch or graph dependencies. `LocalEmbedder` needs the `local` extra:

```sh
uv add itk-core            # OpenAI / Azure OpenAI only
uv add "itk-core[local]"   # plus sentence-transformers
```

Embedding stores default to `itk_core.paths.CACHE_PATH`, created on first use.

Tests: `uv run pytest packages/itk-core/tests`.
