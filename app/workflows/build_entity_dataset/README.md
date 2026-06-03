# Build Entity Dataset

## Overview

**Build Entity Dataset** discovers and extracts structured information about real-world entities from the web, producing a citable, schema-aligned dataset that can be explored through a downloadable interactive web interface.

The workflow performs systematic web search across many query angles, extracts attribute values from authoritative sources, deduplicates and merges records, and verifies attribute values with supporting citations. Every value traces back to at least one real web source.

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
| Model | Yes | OpenAI model to use (default: `gpt-5.4-mini`) |
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
