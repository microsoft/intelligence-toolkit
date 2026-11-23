# itk-entities

Build Entity Dataset for Intelligence Toolkit: research a category of
real-world entities from the web, then curate, audit, recategorize and
release the resulting dataset. The research engine is
[`itk-schemify`](../itk-schemify); this package is the workflow on top of it.

It is a restructuring of `intelligence_toolkit.build_entity_dataset`, whose
single `BuildEntityDataset` class (about 4,000 lines) is split here into stage modules
of plain functions over one explicit `Session`. Behaviour is unchanged and is
checked against the original class by the parity tests described below.

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| State | `itk_entities.state` | `Session` (the live Schemify instance, dataset, progress, usage, worker thread), `ResearchProgress`, `UsageStats`, and read-only views such as `schema_attributes`, `current_dataframe`, `count_unverified`. |
| Storage | `itk_entities.storage` | `build_dataset_json`, throttled `snapshot_partial`, `save_run`, `list_saved_runs`, `load_saved_run`, `import_dataset`, JSON and CSV bytes. |
| Inputs | `itk_entities.inputs` | `parse_candidate_file` and `reference_labels_from_file` for uploaded name lists. |
| Research | `itk_entities.research` | `start_research`, `continue_research`, `stop_research`, `append_guidance`, `add_candidate_entities`, `propose_search_languages`. |
| Unattended runs | `itk_entities.auto_runs` | `start_auto_mode` and `start_rediscovery_benchmark`, built on the loop in `itk_entities.auto`. |
| Refinement | `itk_entities.refine` | `start_verification`, `start_completion`, `start_normalize`. |
| Review | `itk_entities.review` | Verify proposed corrections against fresh public sources; never applies them. |
| Exclusions | `itk_entities.exclusions` | Exclusion rules by label or attribute value. |
| Schema | `itk_entities.schema` | Add, rename, update and remove attributes. |
| Curation | `itk_entities.curation` | Aliases, canonical names, merges, splits, removals, restores, alias suggestions. |
| Audits | `itk_entities.audits` | Harmful-content scan, scope, relevance, merge-quality and duplicate audits, and applying their results. |
| Recategorization | `itk_entities.recategorize` | Propose and apply a taxonomy, fill gaps by search, relevance filter. |
| Value edits | `itk_entities.value_edits` | Set values, pick a single value, merge variants, rename a value. |
| Release | `itk_entities.release` | Release CSV, manifest, dashboard bundle, theme handling. |

Modules depend in one direction: `state` ← `storage` ← every other stage.

## Usage

`BuildEntityDataset` holds a `Session` and forwards each call to its stage
module. It has the same public methods and signatures as the original class.

```python
from itk_entities import BuildEntityDataset

builder = BuildEntityDataset()  # or BuildEntityDataset(runs_dir=Path("runs"))
builder.start_research(api_key="sk-...", category="Open-source relational databases")
while builder.is_running:
    builder.refresh_progress()
    print(builder.progress.stage, builder.progress.entity_count)

builder.add_label_exclusion("Not a database", reason="out of scope")
csv_bytes = builder.get_release_csv_bytes()
```

The stage functions can also be called directly on a session:

```python
from itk_entities import Session, curation, storage

session = Session()
storage.load_saved_run(session, "runs/20260101-120000_tools/data.json", api_key="sk-...")
curation.merge_records(session, "ACME TRACKER", "ACME TRACKER PRO")
```

## Differences from the original class

- Saved runs go to `Session.runs_dir`, set with `BuildEntityDataset(runs_dir=...)`.
  The default is the same cache folder as before. `list_saved_runs` takes an
  optional `runs_dir`.
- State that was private on the class (`_schemify`, `_dataset_json`, ...) is on
  `builder.session` without the underscore.
- It runs on `itk-schemify`, so it has that package's fixes: the budget cap
  stops a run, near-miss names reach the model for duplicate checking, prompts
  with literal braces are sent as written, and completion does not retry a
  cached page that filled nothing.

## Tests

```bash
uv run pytest packages/itk-entities/tests
```

- `test_entities_legacy_parity.py` runs the same 74-step script of non-model
  operations against the original class and this package, comparing every
  return value and the dataset after each step.
- `test_entities_legacy_run_parity.py` does the same for the background
  workers (research, completion, verification, normalization, auto mode,
  rediscovery benchmark) with Schemify's network steps replaced identically on
  both sides.
- `test_entities_legacy_audit_parity.py` does the same for the model-assisted
  audits and recategorization with canned model replies.
- The other four files are the original workflow tests, repointed at this
  package and `itk-schemify`.

The parity tests are skipped when `intelligence_toolkit` is not importable.
