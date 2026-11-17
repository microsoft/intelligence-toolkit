# itk-schemify

Schemify is the engine behind the Build Entity Dataset workflow: it discovers
entities in a category from public web sources, proposes a schema for them,
extracts cited attribute values, resolves duplicates, and audits the result.

This package is a standalone copy of the `intelligence_toolkit.schemify` module,
with four fixes listed below. It does not depend on the rest of Intelligence Toolkit.
It calls OpenAI through its own `itk_schemify.llm` client; moving that onto
`itk-core` is planned as a separate change. The app and
`intelligence_toolkit.build_entity_dataset` still use the original module.

```python
from itk_schemify import Schemify, SchemifyConfig

schemify = Schemify(SchemifyConfig(api_key="sk-..."))
await schemify.initialize(
    category="Open-source relational databases",
    guidance="Focus on widely deployed systems with active communities",
)
```

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
