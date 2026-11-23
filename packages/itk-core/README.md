# itk-core

Shared AI core for Intelligence Toolkit workflows:

- `itk_core.client` / `itk_core.base_chat` / `itk_core.utils`: OpenAI and Azure OpenAI chat, with transient-only retry, streaming and bounded concurrency. `OpenAIClient.generate_response` / `generate_response_async` send a Responses API request under the same retry policy and return the SDK response.
- `itk_core.openai_embedder` / `itk_core.local_embedder`: embedders backed by a LanceDB `VectorStore` cache.
- `itk_core.errors`: typed `LLMError` and `EmbeddingError`.
- `itk_core.usage`: token usage of every chat and embedding call, reported to callbacks you register.

It has no Streamlit, torch or graph dependencies, and `LocalEmbedder` needs the `local` extra (sentence-transformers).

It is not published on its own: the `intelligence-toolkit` wheel bundles `itk_core`, so `pip install intelligence-toolkit` provides it. (The PyPI name `itk-core` belongs to the unrelated ITK image library.) To use it alone from a checkout:

```sh
uv pip install ./packages/itk-core            # OpenAI / Azure OpenAI only
uv pip install "./packages/itk-core[local]"   # plus sentence-transformers
```

To count tokens across a workflow, register a callback for the process:

```python
from itk_core.usage import LLMUsage, usage_callback

calls: list[LLMUsage] = []
with usage_callback(calls.append):
    ...  # run any workflow
total_tokens = sum(call.total_tokens or 0 for call in calls)
```

A count the provider does not report is `None`, not zero. While a callback is registered, streamed chat calls ask for usage with `stream_options`, which needs an API version that accepts it.

Embedding stores default to `itk_core.paths.CACHE_PATH`, created on first use.

Tests: `uv run pytest packages/itk-core/tests`.
