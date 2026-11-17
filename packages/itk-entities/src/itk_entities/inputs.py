# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Parsing uploaded files into entity names: reference lists and candidate lists."""

from __future__ import annotations

import io
import json
import logging

import pandas as pd

_logger = logging.getLogger(__name__)


def reference_labels_from_file(filename: str, raw: bytes) -> list[str]:
    """Parse an uploaded reference dataset and return the entity labels.

    Accepts the same formats as :meth:`parse_candidate_file` and
    additionally recognises the full dataset JSON shape written by
    :meth:`_save_run` (``{"records": [{"label": ...}, ...]}``).
    Duplicates are deduplicated case-insensitively while preserving
    input order.
    """
    if not raw:
        return []
    name = (filename or "").lower()
    if name.endswith(".json"):
        try:
            text = raw.decode("utf-8-sig", errors="replace")
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            names = []
            for r in data["records"]:
                if isinstance(r, dict):
                    v = r.get("label") or r.get("name")
                    if isinstance(v, str) and v.strip():
                        names.append(v.strip())
            return dedupe_preserving_order(names)

    names = parse_candidate_file(filename, raw)
    return dedupe_preserving_order(names)


def dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        v = (raw or "").strip()
        if not v:
            continue
        k = v.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def parse_candidate_file(filename: str, raw: bytes) -> list[str]:
    """Best-effort extraction of candidate names from an uploaded file.

    Supports plain text (one name per line or comma-separated), CSV
    (first column or a column whose header contains ``name``/``label``/
    ``entity``), and JSON (list of strings or list of dicts with one of
    those keys).
    """
    if not raw:
        return []
    name = (filename or "").lower()
    try:
        text = raw.decode("utf-8-sig", errors="replace")
    except Exception:  # noqa: BLE001
        text = raw.decode("latin-1", errors="replace")

    if name.endswith(".json"):
        try:
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            return []
        out: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for key in ("label", "name", "entity", "title"):
                        v = item.get(key)
                        if isinstance(v, str) and v.strip():
                            out.append(v)
                            break
        return out

    if name.endswith(".csv") or name.endswith(".tsv"):
        sep = "\t" if name.endswith(".tsv") else ","
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
        except Exception:  # noqa: BLE001
            return [line.strip() for line in text.splitlines() if line.strip()]
        cols = list(df.columns)
        target = cols[0]
        for c in cols:
            lc = str(c).lower()
            if any(k in lc for k in ("label", "name", "entity")):
                target = c
                break
        return [str(v).strip() for v in df[target].dropna().tolist() if str(v).strip()]

    # Plain text fallback: split on commas or newlines.
    return [tok.strip() for tok in text.replace(",", "\n").splitlines() if tok.strip()]
