"""
Schema proposal step for re-categorize-in-place workflow.

Asks an LLM (via the same backend dispatcher as audit.py) to propose tightened
taxonomies and a complete value-translation table for an existing dataset, then
expands the translations into per-record remappings and out-of-scope flags that
`apply_recategorization` can consume directly.

The model never sees per-record data — only per-attribute observed-value
frequency tables. The expansion step is mechanical and deterministic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .audit import _extract_json_object, _run_openai, _run_subagent

PROMPT_FILENAME = "schema_proposal.md"

# Self-contained default prompt so the workflow works inside ITK without
# shipping an external prompts directory. `run_schema_proposal` still prefers
# a `schema_proposal.md` file when a `prompts_dir` is supplied and it exists.
DEFAULT_SCHEMA_PROPOSAL_PROMPT = """# Schemify Schema Proposal Prompt

Propose tightened, shorter taxonomies for an existing Schemify dataset — without re-running web search. The model reviews the observed-value frequency tables and the user's constraints, then proposes a revised schema and a **complete value translation table** for every observed value. Per-record remappings are then expanded mechanically from that table.

This is step 1 of the **re-categorize in place** workflow. The output is reviewed by a human and applied by the recategorize step.

---

## Inputs

- **constraints** (free text) — what the user wants tightened.
- **observed_values** — per-attribute frequency tables: `{attr: [{value, count}, ...]}`. Every value in these tables MUST appear as a key in the corresponding `value_translations` sub-object — no abbreviations, no omissions.

## Output contract

Return a single JSON object with these keys:

```json
{
  "schema": [
    {
      "name": "attribute_name",
      "description": "...",
      "is_closed_set": true,
      "is_multi_value": true,
      "canonical_values": ["...", "..."],
      "change_summary": "Reduced from N to M values by …"
    }
  ],
  "added_attributes": [
    {"name": "tool_description", "description": "...", "is_closed_set": false, "is_multi_value": false}
  ],
  "removed_attributes": ["Key Phrases"],
  "target_users_grouping": {
    "Parent": ["Canonical child 1", "Canonical child 2"]
  },
  "value_translations": {
    "attribute_name": {
      "<every observed value>": "<new canonical value>"
    }
  }
}
```

### `value_translations` rules

- **Every** observed value in the input frequency table must appear as a key.
- Map each old value to:
  - a string from your proposed `canonical_values` (the normal case), or
  - `null` to **drop** that value (e.g. attribute removed, or value falls into a dropped category), or
  - `"UNRESOLVED"` if you cannot confidently assign it — the human reviews these.
- `removed_attributes` do not need a `value_translations` entry — the whole attribute is dropped.

## Instructions

1. **Cluster** observed values by meaning, atomicity (one dimension per value), and frequency. Collapse synonyms and one-offs into a small canonical set. Preserve discriminating power — don't over-merge.
2. **Emit a complete `value_translations` table.** Walk each observed value and map it explicitly. This is the load-bearing artifact; per-record remappings are derived from it mechanically.
3. **`change_summary`** — one sentence per attribute explaining what got merged or dropped.
4. **Sibling groupings** — for attributes flagged for grouping, emit `target_users_grouping`: parent → list of canonical children.
5. **Added attributes** — propose name, description, closed-set/multi-value. For closed-set added attributes, also provide `canonical_values`.

## Constraints

- Use only the observed-value tables provided. Do not invent values that aren't in the input.
- Confidence ≥ 0.7 implied for non-`UNRESOLVED` mappings.
- Output **only** the JSON object. No prose, no markdown fences, no tool calls.
"""


def _load_prompt(prompts_dir: Path | None) -> str:
    """Return the schema-proposal prompt.

    Prefers `<prompts_dir>/schema_proposal.md` when supplied and present;
    otherwise falls back to the embedded default so the workflow is
    self-contained inside ITK.
    """
    if prompts_dir is not None:
        candidate = prompts_dir / PROMPT_FILENAME
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return DEFAULT_SCHEMA_PROPOSAL_PROMPT


def _observed_values(data: dict, target_attrs: list[str]) -> dict[str, list[dict]]:
    freq: dict[str, Counter] = {a: Counter() for a in target_attrs}
    for r in data.get("records", []):
        for a in target_attrs:
            for v in r.get("attributes", {}).get(a, {}).get("values", []):
                if v.get("value"):
                    freq[a][v["value"]] += 1
    return {a: [{"value": v, "count": c} for v, c in freq[a].most_common()] for a in target_attrs}


def _build_message(prompt: str, constraints: str, observed: dict) -> str:
    return (
        prompt
        + "\n\n---\n\n## Inputs\n\n### constraints\n\n"
        + constraints
        + "\n\n### observed_values\n\n```json\n"
        + json.dumps(observed, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def _normalize_proposal(proposal: dict) -> dict:
    """Coerce common model variations to the strict schema."""
    ra = proposal.get("removed_attributes") or []
    proposal["removed_attributes"] = [x["name"] if isinstance(x, dict) else x for x in ra]
    return proposal


def _validate_translation_coverage(observed: dict, translations: dict) -> list[str]:
    issues: list[str] = []
    for attr, entries in observed.items():
        translated = set((translations.get(attr) or {}).keys())
        missing = {e["value"] for e in entries} - translated
        if missing:
            issues.append(f"{attr}: {len(missing)} observed values missing from value_translations")
    return issues


def _expand_remappings(
    data: dict, proposal: dict, target_attrs: list[str]
) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk records and convert value_translations into per-record remappings.

    Returns (record_remappings, out_of_scope_records, unresolved).

    A record is flagged out_of_scope if any target attribute had values that ALL
    mapped to null — i.e. the dropped categories on that attribute leave nothing
    behind to categorize the record by.
    """
    vt = proposal.get("value_translations") or {}
    remaps: list[dict] = []
    out_of_scope: list[dict] = []
    unresolved: list[dict] = []

    for r in data.get("records", []):
        label = r.get("label")
        per_attr_new: dict[str, set] = {a: set() for a in target_attrs}
        per_attr_had: dict[str, bool] = {a: False for a in target_attrs}
        for a in target_attrs:
            for v in r.get("attributes", {}).get(a, {}).get("values", []):
                old = v.get("value")
                if old is None:
                    continue
                if a not in vt or old not in vt[a]:
                    unresolved.append({
                        "label": label, "attribute": a,
                        "issue": f"old value {old!r} not in translation table",
                    })
                    continue
                per_attr_had[a] = True
                new = vt[a][old]
                if new == "UNRESOLVED":
                    unresolved.append({
                        "label": label, "attribute": a,
                        "issue": f"UNRESOLVED translation for {old!r}",
                    })
                elif new is None or new != old:
                    remaps.append({
                        "label": label, "attribute": a,
                        "old_value": old, "new_value": new,
                        "confidence": 0.85, "reason": "value_translations",
                    })
                    if new is not None:
                        per_attr_new[a].add(new)
                else:
                    per_attr_new[a].add(new)
        # Out-of-scope: any attr where all values were dropped
        for a in target_attrs:
            if per_attr_had[a] and not per_attr_new[a]:
                out_of_scope.append({
                    "label": label,
                    "reason": f"All `{a}` values mapped to dropped categories.",
                    "recommended_action": "remove",
                    "trigger_attribute": a,
                })
                break  # one out_of_scope entry per record is enough

    return remaps, out_of_scope, unresolved


def run_schema_proposal(
    data: dict,
    constraints: str,
    target_attrs: list[str] | None,
    prompts_dir: Path | None = None,
    backend: str = "openai",
    model: str = "gpt-5.2",
    api_key: str | None = None,
    subagent_command: str | list[str] | None = None,
) -> dict[str, Any]:
    """Propose a tightened schema and expand value_translations into remappings.

    `target_attrs` defaults to every closed-set attribute in the dataset.
    `prompts_dir` is optional; when omitted (or the file is missing) the
    embedded default prompt is used.
    """
    if target_attrs is None:
        target_attrs = [a["name"] for a in data.get("schema_attributes", [])
                        if a.get("is_closed_set")]

    prompt = _load_prompt(prompts_dir)
    observed = _observed_values(data, target_attrs)
    message = _build_message(prompt, constraints, observed)

    if backend == "openai":
        text = _run_openai(message, model=model, api_key=api_key)
    elif backend == "subagent":
        text = _run_subagent(message, command=subagent_command)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    try:
        proposal = _normalize_proposal(_extract_json_object(text))
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"backend {backend!r} did not return valid JSON: {e}\n---\n{text[:1000]}"
        )

    issues = _validate_translation_coverage(observed, proposal.get("value_translations") or {})
    proposal.setdefault("validation_issues", issues)

    remaps, oos, unresolved = _expand_remappings(data, proposal, target_attrs)
    proposal["record_remappings"] = remaps
    proposal["out_of_scope_records"] = oos
    proposal["unresolved"] = (proposal.get("unresolved") or []) + unresolved
    return proposal
