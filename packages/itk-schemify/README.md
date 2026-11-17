# itk-schemify

Schemify is the engine behind the Build Entity Dataset workflow: it discovers
entities in a category from public web sources, proposes a schema for them,
extracts cited attribute values, resolves duplicates, and audits the result.

This package is a standalone copy of the `intelligence_toolkit.schemify` module,
with the differences listed below. Its only toolkit dependency is `itk-core`,
whose `OpenAIClient` sends every model request. The app and
`intelligence_toolkit.build_entity_dataset` still use the original module.

```python
from itk_schemify import Schemify, SchemifyConfig

schemify = Schemify(SchemifyConfig(api_key="sk-..."))
await schemify.initialize(
    category="Open-source relational databases",
    guidance="Focus on widely deployed systems with active communities",
)
```

By default requests go to OpenAI with `config.api_key`. To use other connection
settings, pass an `itk_core` `OpenAIConfiguration` as
`Schemify(config, ai_configuration=...)`. Schemify relies on the Responses API
and its `web_search` tool, and has not been run against Azure OpenAI.

## Differences from `intelligence_toolkit.schemify`

- **Budget cap stops the run.** Once the estimated cost passes `max_budget`, no
  further request is sent and `BudgetExceededError` propagates out of discovery,
  completion, verification and the audits. The response that crossed the budget
  is still returned and logged.
- **Near-miss names reach the model.** `find_fuzzy_match` reports its best score
  on a miss, so `check_duplicates` sends labels scoring within 20 points of the
  threshold to the model, and ignores any pair naming a label it was not given.
- **Prompts are formatted once.** A prompt passed without `variables` is sent as
  written, so braces in guidance or entity labels no longer raise.
- **No pointless completion retries.** With caching on, `complete_all` stops
  retrying an entity after a call that filled nothing, since a retry would
  re-read the same cached page.
- **Requests go through `itk-core`.** Transient failures (rate limits, timeouts,
  connection and 5xx errors) are retried up to 4 times with exponential backoff,
  honouring `Retry-After`, instead of up to 5 times on errors whose message
  contained a status code. A request that still fails raises
  `itk_core.errors.LLMError` rather than the OpenAI SDK error.

## Command line

```bash
python -m itk_schemify.cli --help
```

| Command | What it does |
| ------- | ------------ |
| `run` | Run a discovery and extraction job from a JSON config. |
| `audit` | Audit a dataset for quality issues. |
| `propose-schema` | Propose a schema for a dataset. |
| `recategorize` | Apply a schema proposal to a dataset. |
| `dedup-audit` | Propose duplicate merges for review. |
| `dedup-apply` | Apply approved dedup-audit proposals to a copy of the dataset. |
| `verify-review` | Verify proposed corrections against fresh public sources; never applies them. |
| `build-dashboard` | Bundle a shippable dashboard folder. |

## Tests

```bash
uv run pytest packages/itk-schemify/tests
```

Tests that exercise schemify through the Build Entity Dataset API remain in
`intelligence_toolkit/tests/unit/schemify`, against the original module, until
that workflow is migrated.
