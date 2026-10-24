# Build Entity Dataset

## Overview

**Build Entity Dataset** discovers and extracts structured information about real-world entities from the web, producing a citable, schema-aligned dataset that can be explored through a downloadable interactive web interface.

The workflow performs systematic web search across many query angles, extracts attribute values from authoritative sources, deduplicates and merges records, and can verify attribute values with supporting citations. Values derived from general knowledge may be uncited unless verification is enabled.

### Typical uses
- Building a landscape map of technologies, organisations, databases, or tools in a domain
- Creating a structured directory of entities for research or reporting
- Generating grounded reference data for downstream analysis workflows

---

## How it works

### 1. Define task
Specify what category of entities to collect (e.g., *European Union member states*, *Open-source Python web frameworks*) and any guidance that steers extraction quality or scope. You may optionally provide a JSON list of schema attributes to use instead of the auto-proposed schema.

### 2. Run research
The workflow runs structured web search queries in three phases:

| Phase | Purpose |
|-------|---------|
| **Discovery** | Broad queries to find candidate entities |
| **Targeted** | Intersectional queries to fill coverage gaps |
| **Completion** | Attribute-filling for partially complete records |

A live progress display shows queries run, entities found, and current stage. You can stop the research at any time and save partial results.

### 3. Review dataset
Inspect the discovered entities in a tabular view. The schema attributes and their coverage are shown. You can also load a previously saved `data.json` file for review.

#### Verify review decisions against fresh evidence

In **Review dataset**, open **Verify review decisions** to upload a CSV or JSON
queue of proposed repairs, renames, splits, duplicate checks or scope reviews.
This is separate from **Verify attribute values**: it checks proposed corrections
and **never applies them**, including on datasets loaded without an OpenAI key.

Choose OpenAI or the authenticated Copilot CLI installed on the application host,
authorize account usage, optionally select zero-based record IDs, and start a
fresh capture. Copilot uses that host's authenticated account and default model
routing; it does not use the browser user's identity. Proposal/record context and
retrieved public text are sent to the selected AI service. Configure an OpenAI key
in Settings when using OpenAI. Copilot must already be installed and authenticated;
the application does not install it or log in on your behalf.
Reported usage covers recorded completions, not an authoritative billing total;
interrupted requests may still incur account usage.

The verifier separately checks the primary identity, proposed description,
**every original alias**, each split candidate, each related-record overlap and
each outstanding prerequisite. Results are **supported**, **contradicted** or
**inconclusive**, with exact source quotations, retrieval dates, hashes and errors.
Duplicate checks receive both records' context and source candidates; partial
overlap is not permission to merge bundled records. Scope checks use the dataset's
existing inclusion guidance.

- **Supported** means the retrieved evidence supports that check, not that a tool
  is effective, high quality, independently corroborated, or approved for publication.
- Matching titles, navigation, shared domains and historical AI summaries do not
  constitute substantive fresh evidence.
- Failed, browser-only, oversized or unsupported sources remain explicit.
  This is linked-source retrieval, **not an exhaustive search of the web**.
  Raise the source limit to cover more candidates; omitted URLs are listed.
- Overall record status includes outstanding human/workflow prerequisites.
  **Factual status** excludes those prerequisites. A supported identity or
  description does not establish all alias, split, scope or duplicate claims.
  **Contradicted** means at least one check conflicts with fresh evidence, not
  necessarily that the original triage action was wrong. An open alias
  relationship can become resolved, or an inclusion check can establish an
  exclusion. Related products, components and capabilities are not automatically
  interchangeable tool names; read the per-check reasons.
- Pause waits for in-flight requests and preserves checkpoints. Opening or
  resuming a saved capture does not relabel its evidence as newly retrieved.
  Start a **new** capture to refresh all pages. Alternatively, explicitly
  **retry failed sources on resume**: successful snapshots retain their dates,
  new attempts get new dates, prior failures are retained, and affected
  records are reassessed when their available evidence changes.
- Saved captures are listed for the same dataset fingerprint. Reports remain
  separate from dataset files; no normalization, merges, splits, alias changes or
  exclusions run as part of verification. A changed dataset makes an open or
  imported report stale.

Download the record-level decisions CSV, per-check CSV, JSON report, or evidence
ZIP containing extracted source snapshots, manifest, proposals and results.
An exported JSON report can also be reopened in Review; retain the ZIP for the
underlying text snapshots and resumable per-record checkpoints. Keep the original
input dataset alongside the capture. Human approval and any later application
are separate.

**Proposal format.** JSON accepts an array or `{"items": [...]}`. Each row requires
`record_id` (or `id`), the exact `current_label`, and `action` (or
`recommended_action`): `repair_record`, `rename_record`, `split_record`,
`possible_duplicate`, `scope_review`, or `research_needed`. Optional fields:
`proposed_label`, `proposed_description`, `rationale`, `references`,
`checks_before_apply`, `split_entities`, and `related_record_ids`.
`alias_decisions` must contain every original alias exactly once with an `alias`
and a `decision` of `keep`, `detach` or `verify`. No aliases are silently truncated.
Use JSON arrays for names containing newlines. Split entities may be names or
objects with a `label`; splits need at least two candidates. Duplicate proposals
need related IDs. Optional `related_record_labels` maps string IDs to exact
canonical labels. An envelope `dataset_fingerprint` binds the initial import to
an exact dataset; otherwise initial binding checks canonical IDs/labels and alias
coverage, not the age of every historical field.

CSV also accepts newline-separated `aliases_to_keep`, `aliases_to_detach`,
`aliases_to_verify`, `split_entities`, `references` and `checks_before_apply`,
comma-separated `related_record_ids`, and optional `related_records` lines of
`ID: canonical label`. JSON arrays inside CSV cells are also supported for list
fields. Existing first-pass triage CSV/JSON exports are accepted.

**CLI** (from an environment with ITK installed):

```powershell
python -m intelligence_toolkit.schemify.cli verify-review data.json `
  --proposals triage.csv --out review_evidence --backend copilot `
  --concurrency 4 --max-sources 48
```

Use `--backend openai` with `OPENAI_API_KEY`; `--model` applies only to OpenAI.
Use `--record-ids 4,21,75` to select a subset. Explicit `--resume` requires the
same dataset, normalized proposals, backend/model and source limit and reuses the
capture's original timestamps. CLI JSON imports also check a matching
`source_hashes` entry for the dataset filename when supplied.
Add `--retry-failed-sources` to `--resume` to retry inaccessible URLs without
re-fetching successful snapshots or silently reusing decisions after evidence
changes. Previous failed attempts remain in source metadata.
Exit code `2` means the report contains record processing/retrieval errors, not
that every inconclusive check is an execution failure. Checkpoints are retained.

**Library:**

```python
from pathlib import Path
from intelligence_toolkit.schemify.review_verification import (
    load_proposals, make_client, verify_proposals,
)

proposals = load_proposals(Path("triage.csv").read_text(encoding="utf-8-sig"), "csv")
report = await verify_proposals(
    dataset, proposals, Path("review_evidence"), make_client("copilot"),
    backend="copilot", max_sources=48,
)
```

`BuildEntityDataset.start_review_verification` exposes the same implementation
as a background job. Read `review_verification_report` and
`review_verification_progress`; use `stop_review_verification`,
`list_review_verifications`, `open_review_verification`,
`resume_review_verification` and `review_verification_bundle` to pause,
reopen, resume and export without altering the dataset.

### 4. Export
Download the dataset as:
- **`data.json`** — full structured dataset with per-value source citations
- **`data.csv`** — flat CSV for use in spreadsheets or other workflows

Or build a **themed web interface**: supply a title, subtitle, colours, and an optional logo, then download a self-contained HTML dashboard that anyone can open in a browser to search and filter the dataset — no server required.

---

## Input requirements

| Field | Required | Description |
|-------|----------|-------------|
| Entity category | Yes | Short description of what entities to find |
| Guidance | No | Free-text instructions on scope, quality, or exclusions |
| Schema attributes | No | JSON list of `{name, description, is_closed_set}` objects |
| Model | Yes | OpenAI model to use (default: `gpt-6-luna`) |
| Max queries | Yes | Total web-search budget (default: 30) |
| Budget (USD) | Yes | Maximum LLM/search spend (default: $10) |

---

## Tutorial: European countries

This example uses the category **"Countries of Europe"** with guidance *"Include all sovereign states with territory on the European continent."*

### Step 1 — Define task
- **Category:** `Countries of Europe`
- **Guidance:** `Include all sovereign states with territory on the European continent.`
- Leave schema blank to let the model propose one.

### Step 2 — Run research
Click **Start research**. The workflow will:
1. Propose a schema (e.g., capital, population, area, EU membership, currency, official language, government type)
2. Run broad discovery queries ("European countries", "sovereign states in Europe")
3. Run targeted queries per attribute value ("countries in Europe using the Euro")
4. Verify and fill missing values

Typical run: ~30 queries, ~2–3 minutes, <$1 with `gpt-5.4-mini`.

### Step 3 — Review dataset
The dataset table will show one row per country with columns for each schema attribute.

### Step 4 — Export
Download `data.csv` for analysis, or build a web interface with a title such as *"Countries of Europe"*, your institution's colours, and optionally a logo. Extract the ZIP and open `dashboard/dashboard.html` to explore the interactive map.

---

## Output format

### data.json
```json
{
  "category": "Countries of Europe",
  "guidance": "...",
  "schema_attributes": [
    {"name": "capital", "description": "Capital city", "is_closed_set": false},
    ...
  ],
  "records": [
    {
      "label": "France",
      "aliases": ["French Republic"],
      "attributes": {
        "capital": {
          "values": [
            {
              "value": "Paris",
              "sources": [{"url": "https://...", "title": "...", "tier": "AUTHORITATIVE"}]
            }
          ],
          "confidence": 0.95,
          "verified": true
        }
      }
    }
  ]
}
```

### data.csv
A flat CSV with one row per entity and one column per schema attribute. Multi-valued attributes are joined with `; `.

---

## Responsible use

- **Verify important values** — AI extraction may introduce errors. Review outputs before acting on them.
- **Respect source terms of service** — web search uses the OpenAI web search tool; usage is governed by OpenAI's usage policies.
- **Cost awareness** — larger query budgets and more capable models increase cost. Set a billing cap in your OpenAI account.
- **No personal data** — do not use this workflow to collect personal information about individuals.
