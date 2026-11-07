# itk-extract

Structured record extraction for Intelligence Toolkit, as a pipeline of pure
functions from unstructured text to polars frames:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1 | `itk_extract.sources` | Builds `Source`s from plain texts or from the rows of a frame. |
| 2 | `itk_extract.chunk` | Optionally splits long sources into token-bounded `Chunk`s (itk-core splitter). |
| 3 | `itk_extract.schema` | `RecordSchema`: validates the JSON schema, lists its record arrays, builds the strict OpenAI response format. |
| 4 | `itk_extract.prompt` | System/user messages for one chunk, with optional guidance. |
| 5 | `itk_extract.extract` | One model call per distinct chunk text; replies parsed and validated. Bad replies are recorded, not raised. |
| 6 | `itk_extract.merge` | Merges per-chunk objects; arrays concatenate with `source_id`/`chunk_id` provenance, scalar disagreements become `Conflict`s. |
| 7 | `itk_extract.records` | One flat frame per record array, provenance columns last, optional dedup. |

`itk_extract.Extractor` strings these together and keeps the per-chunk
extractions on the `ExtractResult`, so the merge policy can be changed with
`remerge` without calling the model again.

```python
import json

import polars as pl
from itk_core.openai_configuration import OpenAIConfiguration
from itk_extract import ExtractConfig, Extractor, RecordSchema, from_frame, from_texts

schema = RecordSchema(json.load(open("people_schema.json")))

# Either one long document, split into ~1000-token chunks...
sources = from_texts([open("report.txt").read()])
config = ExtractConfig(chunk_tokens=1000, guidance="Use ISO dates.")

# ...or one source per CSV row, as the Streamlit workflow does.
sources = from_frame(pl.read_csv("rows.csv"), id_column="id")
config = ExtractConfig()

extractor = Extractor.from_openai(OpenAIConfiguration())
result = await extractor.extract(sources, schema, config)
result.object  # merged JSON object
result.frames["people"]  # polars frame with source_id / chunk_id columns
result.conflicts_frame  # scalar fields that differed across chunks
result.errors_frame  # chunks whose reply could not be used

by_majority = Extractor.remerge(
    result, config.model_copy(update={"merge_policy": "majority"})
)
```

Only the extract stage calls an external API. The package has no pandas or
Streamlit dependency and, like `itk-core` and `itk-match`, is bundled into the
`intelligence-toolkit` wheel rather than published on its own.

Tests: `uv run pytest packages/itk-extract/tests`.
