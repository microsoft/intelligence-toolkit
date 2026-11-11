# itk-mock

Mock data generation for Intelligence Toolkit: describe records with a JSON
schema, have a model generate records that follow it, then have it write one
text per record. Each stage is a module of plain functions:

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1 | `itk_mock.schema` | Builds and edits the JSON schema in place: `create_boilerplate_schema`, `add_object_field` / `add_array_field` / `add_primitive_field`, rename, delete, move, required / enum / additional-properties status, string and number constraints, `normalize_schema_for_openai`, and `extract_array_fields` (the first array is the primary record array). Every edit addresses a field by its `nesting` path and label. |
| 2 | `itk_mock.template` | `generate_object_from_schema` (the emptiest object a schema allows) and `evaluate_schema` / `evaluate_object_and_schema`, which always return a `ValidationResult`. |
| 3 | `itk_mock.chat` | The `ChatFn` protocol: one async call taking messages, an optional JSON response schema and a temperature. `openai_chat` adapts itk-core's `OpenAIClient` and is the only place that imports it, lazily. |
| 4 | `itk_mock.prompts` | The workflow's three prompts, word for word, and their message builders. |
| 5 | `itk_mock.records` | `generate_records`: one unseeded batch, then rounds of batches seeded from sampled existing records with near-duplicate and close-relation targets. Also `merge_json_objects`, `sample_from_record_array` and `select_random_records`, all taking an explicit `random.Random`. |
| 6 | `itk_mock.frames` | `to_frames`: one polars frame per record array, keyed by its dotted path. Nested objects become `outer.inner` columns; arrays stay as list columns and also get their own frame. |
| 7 | `itk_mock.texts` | `generate_texts`: one text per record, `parallelism` at a time, in input order. |

`itk_mock.MockDataGenerator` strings these together. `generate_records` returns
a `MockResult` holding the merged JSON object, every parsed model reply and the
per-array frames, so `generate_texts` runs later on those frames (or on any
loaded CSV) without generating records again.

```python
import json
from pathlib import Path

from itk_mock import MockConfig, MockDataGenerator, PrimitiveFieldType, TextConfig
from itk_mock import schema as sb
from itk_mock.template import evaluate_schema

schema = sb.create_boilerplate_schema(title_field="Customers")
sb.add_primitive_field(schema, ["records"], "name", "Full name")
sb.add_primitive_field(schema, ["records"], "age", field_type=PrimitiveFieldType.NUMBER)
sb.set_number_minimum(schema, ["records"], "age", 18)
evaluate_schema(schema)               # ValidationResult.VALID

# or load one, closing objects and requiring every property for structured outputs
schema = sb.normalize_schema_for_openai(json.loads(Path("schema.json").read_text()))

generator = MockDataGenerator.from_openai()      # OPENAI_API_KEY etc. from the environment
result = await generator.generate_records(
    schema,
    MockConfig(
        num_records_overall=100,
        records_per_batch=10,
        duplicate_records_per_batch=1,
        related_records_per_batch=1,
        seed=42,
    ),
)
result.json_object                    # merged object, as the workflow downloads it
result.frames["records"]              # polars frame per record array
result.conflicts                      # scalar paths that differed between batches

texts = await generator.generate_texts(result.primary_frame, TextConfig(parallelism=10))
texts.frame                           # one `mock_text` column, aligned with texts.input_texts
```

Any async callable with the `ChatFn` signature can replace OpenAI; the tests
use a canned-reply fake, so the unit suite needs no network or API key:

```python
async def chat(messages, response_schema=None, temperature=None) -> str: ...

generator = MockDataGenerator(chat)
```

`seed` (or an `rng` passed to `generate_records`) fixes which existing records
seed later batches. With a deterministic chat function the whole run is then
repeatable; with a real model it fixes only the seed-record choice.

## Differences from the legacy workflow module

- **Explicit randomness.** Sampling takes a `random.Random` or `MockConfig.seed`
  instead of the global `random` state.
- **Typed schema edit errors.** Edit functions take `(schema, nesting, label)`
  and raise `SchemaPathError`, `FieldNotFoundError`, `DuplicateFieldError` or
  `FieldTypeError` (all `SchemaEditError`) where the legacy functions skipped
  unknown path labels or overwrote fields. `set_additional_field_status` reports
  whether it changed anything.
- **Labels.** A label passed to an `add_*` function is used as given; only an
  omitted label gets the legacy numbered default (`string_1`, counted across
  the whole schema), bumped if that label is still present where it is added,
  so a delete can no longer lead to an overwrite. New fields are required unless `required=False`, and
  `rename_field` keeps the field's required status. `set_enum_values` is new.
- **Polars frames** instead of pandas. Columns follow the field order of the
  records (pandas moved flattened `outer.inner` columns to the end), and an
  array of primitives gets a `value` column rather than `0`.
- **Deterministic merge.** Keys keep first-seen order and conflicts are kept on
  the result.
- **Batch accounting.** Rounds count the batches actually generated, so a run
  whose first batch has fewer records than the batches still needed keeps going
  until the target is met instead of stopping short.
- **Fully async,** including the first, unseeded call.
- **Validation is total.** `evaluate_object_and_schema` returns
  `OBJECT_INVALID` for every object that fails validation. The facade rejects a
  schema that is not a JSON Schema before calling the model.
- **Errors.** A reply that is not a JSON object, a schema without arrays and an
  empty first batch raise `MockDataError` (a `ValueError`).
- **Text inputs** are compact `json.dumps` output rather than pandas
  `to_json`, so `/` is not escaped.

The package has no pandas or Streamlit dependency and, like `itk-core`, is
bundled into the `intelligence-toolkit` wheel rather than published on its own.

## Example notebook

Open [`examples/generate_mock_data/generate_mock_data.ipynb`](examples/generate_mock_data/generate_mock_data.ipynb)
for a walkthrough of the public API. It loads and validates the existing
[`customer_complaints_schema.json`](../../example_outputs/generate_mock_data/customer_complaints/customer_complaints_schema.json),
builds a small schema from scratch with the builder functions, shows the
template object, and runs record and text generation through an offline,
deterministic chat function defined in the notebook, so it executes without an
API key. A marked cell at the end shows how to swap in the real OpenAI client.
The notebook's first section has the kernel setup; outputs are stripped.

## Tests

```bash
uv run --frozen --extra dev pytest packages/itk-core/tests packages/itk-match/tests packages/itk-extract/tests packages/itk-query/tests packages/itk-networks/tests packages/itk-anonymize/tests packages/itk-compare/tests packages/itk-mock/tests
```

Run the package suites in one invocation, as above; the test directory has no
`__init__.py` and shares helpers through `itk_mock_testing.py`.
