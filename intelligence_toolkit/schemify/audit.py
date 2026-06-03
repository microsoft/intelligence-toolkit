"""
Dataset audit for Schemify.

Runs the audit prompt (prompts/audit.md) against a dataset JSON via the OpenAI
Responses API and returns a structured report (duplicates, offensive content,
mis-categorization, completeness gaps, source weakness).

The CLI wrapper in `schemify.cli` writes both audit.json and audit.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_FILENAME = "audit.md"


def load_prompt(prompts_dir: Path) -> str:
    return (prompts_dir / PROMPT_FILENAME).read_text(encoding="utf-8")


@dataclass
class AuditResult:
    report: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.report, indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        r = self.report
        out: list[str] = ["# Dataset Audit Report", ""]
        s = r.get("summary", {})
        if s:
            out.append("## Summary")
            for k, v in s.items():
                out.append(f"- **{k}**: {v}")
            out.append("")

        def section(title: str, key: str, fmt) -> None:
            items = r.get(key) or []
            out.append(f"## {title} ({len(items)})")
            if not items:
                out.append("_None flagged._\n")
                return
            for it in items:
                out.append(fmt(it))
            out.append("")

        section(
            "Duplicates",
            "duplicates",
            lambda it: f"- **{' / '.join(it.get('labels', []))}** "
            f"(confidence {it.get('confidence', 0):.2f}, "
            f"{it.get('recommended_action', '?')}): {it.get('reason', '')}",
        )
        section(
            "Offensive content",
            "offensive",
            lambda it: f"- **{it.get('label')}** [{it.get('severity', '?')}/"
            f"{it.get('category', '?')}] in `{it.get('field', '?')}`: "
            f"\"{it.get('excerpt', '')}\" → {it.get('recommended_action', '?')}",
        )
        section(
            "Mis-categorized",
            "miscategorized",
            lambda it: f"- **{it.get('label')}** `{it.get('attribute')}`: "
            f"`{it.get('current_value')}` → `{it.get('suggested_value')}` "
            f"({it.get('reason', '')})",
        )
        section(
            "Out of scope",
            "out_of_scope",
            lambda it: f"- **{it.get('label')}** "
            f"({it.get('recommended_action', '?')}): {it.get('reason', '')}",
        )
        section(
            "Completeness gaps",
            "completeness_gaps",
            lambda it: f"- **{it.get('label')}** "
            f"({it.get('pct_empty', 0):.0%} empty): "
            f"{', '.join(it.get('empty_attributes', []))}",
        )
        section(
            "Source weakness",
            "source_weakness",
            lambda it: f"- **{it.get('label')}** "
            f"({it.get('source_count', 0)} sources, top-tier "
            f"{it.get('top_tier', '?')}): {it.get('reason', '')}",
        )
        return "\n".join(out)


def build_user_message(prompt_text: str, data: dict, policy: str | None) -> str:
    parts = [prompt_text, "", "---", "", "## Inputs", ""]
    if policy:
        parts += [f"### policy\n\n{policy}\n"]
    parts += [
        "### data.json",
        "",
        "```json",
        json.dumps(data, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(parts)


def _extract_json_object(text: str) -> dict:
    """Pull the first top-level JSON object out of `text` (handles fenced code blocks)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in output")
    return json.loads(s[start : end + 1])


def _run_openai(message: str, model: str, api_key: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model,
        input=message,
        response_format={"type": "json_object"},
    )
    return getattr(response, "output_text", None) or ""


def _run_subagent(message: str, command: str | list[str] | None) -> str:
    """Pipe `message` to a subagent CLI on stdin; return stdout.

    `command` defaults to `$SCHEMIFY_SUBAGENT_CMD` or `claude -p` — any CLI that
    reads a prompt from stdin and writes the response to stdout works.
    """
    import shlex
    import subprocess

    cmd = command or os.environ.get("SCHEMIFY_SUBAGENT_CMD") or "claude -p"
    argv = cmd if isinstance(cmd, list) else shlex.split(cmd)
    proc = subprocess.run(
        argv, input=message, capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subagent {argv!r} exited {proc.returncode}: {proc.stderr[:500]}"
        )
    return proc.stdout


def run_audit(
    data: dict,
    prompts_dir: Path,
    policy: str | None = None,
    backend: str = "openai",
    model: str = "gpt-5.2",
    api_key: str | None = None,
    subagent_command: str | list[str] | None = None,
) -> AuditResult:
    """Run the audit prompt against a dataset and return the parsed report.

    backend:
      - "openai": direct OpenAI Responses API call (uses `model`, `api_key`).
      - "subagent": pipe the prompt to a model-agnostic subagent CLI on stdin
        (default `claude -p`, override via `subagent_command` or
        `$SCHEMIFY_SUBAGENT_CMD`).
    """
    prompt = load_prompt(prompts_dir)
    message = build_user_message(prompt, data, policy)

    if backend == "openai":
        text = _run_openai(message, model=model, api_key=api_key)
    elif backend == "subagent":
        text = _run_subagent(message, command=subagent_command)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    try:
        report = _extract_json_object(text)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"backend {backend!r} did not return valid JSON: {e}\n---\n{text[:1000]}")
    return AuditResult(report=report)


def apply_recategorization(data: dict, proposal: dict) -> dict:
    """Apply a schema_proposal.json to a dataset in place (returns a new dict).

    Supported proposal keys: removed_attributes, added_attributes (metadata only),
    schema (canonical_values per attribute), record_remappings, out_of_scope_records.
    """
    out = json.loads(json.dumps(data))  # deep copy
    removed = set(proposal.get("removed_attributes") or [])
    schema_updates = {s["name"]: s for s in (proposal.get("schema") or [])}
    out_of_scope = {r["label"] for r in (proposal.get("out_of_scope_records") or [])
                    if r.get("recommended_action") == "remove"}

    # Update schema_attributes
    new_schema = []
    for attr in out.get("schema_attributes", []):
        if attr["name"] in removed:
            continue
        if attr["name"] in schema_updates:
            upd = schema_updates[attr["name"]]
            if "canonical_values" in upd:
                attr["canonical_values"] = upd["canonical_values"]
            if "description" in upd:
                attr["description"] = upd["description"]
        new_schema.append(attr)
    for added in (proposal.get("added_attributes") or []):
        new_schema.append({
            "name": added["name"],
            "description": added.get("description", ""),
            "is_closed_set": added.get("is_closed_set", False),
            "is_multi_value": added.get("is_multi_value", False),
            "canonical_values": added.get("canonical_values", []),
        })
    out["schema_attributes"] = new_schema

    # Build remap index: (label, attribute, old_value) → new_value
    remaps: dict[tuple[str, str, str], str] = {}
    for rm in (proposal.get("record_remappings") or []):
        remaps[(rm["label"], rm["attribute"], rm["old_value"])] = rm["new_value"]

    new_records = []
    for rec in out.get("records", []):
        if rec.get("label") in out_of_scope:
            continue
        attrs = rec.get("attributes", {})
        # Drop removed attributes
        for k in list(attrs.keys()):
            if k in removed:
                del attrs[k]
        # Apply per-value remappings; new_value=None drops that value entry.
        # Dedupe collapsed values within an attribute.
        for attr_name, attr_block in attrs.items():
            kept = []
            seen: set = set()
            for v in attr_block.get("values", []):
                key = (rec["label"], attr_name, v.get("value"))
                if key in remaps:
                    new = remaps[key]
                    if new is None:
                        continue
                    v["value"] = new
                if v.get("value") in seen:
                    continue
                seen.add(v.get("value"))
                kept.append(v)
            attr_block["values"] = kept
        new_records.append(rec)
    out["records"] = new_records
    return out
