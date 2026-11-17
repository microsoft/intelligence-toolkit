# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Release artifacts: CSV, manifest, dashboard bundle and theme handling."""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
from itk_schemify.cli import DASHBOARD_DIR

from . import storage
from .state import Session

_logger = logging.getLogger(__name__)


def get_release_csv_bytes(session: Session) -> bytes:
    """Flat CSV for publication: label, aliases, every schema attribute, sources."""
    data = session.dataset_json or {}
    attrs = [a.get("name", "") for a in data.get("schema_attributes", [])]
    rows = []
    for r in data.get("records", []):
        urls: list[str] = []
        row: dict = {
            "name": r.get("label", ""),
            "aliases": "; ".join(r.get("aliases") or []),
        }
        for attr in attrs:
            vals = (r.get("attributes") or {}).get(attr, {}).get("values", [])
            row[attr] = "; ".join(v.get("value", "") for v in vals if v.get("value"))
            for v in vals:
                for s in v.get("sources") or []:
                    if s.get("url") and s["url"] not in urls:
                        urls.append(s["url"])
        row["source_count"] = len(urls)
        row["sources"] = " | ".join(urls)
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode() if rows else b""


def build_release_manifest(
    session: Session, *, title: str, version: str, notes: str = ""
) -> dict:
    data = session.dataset_json or {}
    records = data.get("records", [])
    schema = data.get("schema_attributes", [])
    value_counts = {}
    for a in schema:
        if not a.get("is_closed_set"):
            continue
        counts: dict[str, int] = {}
        for r in records:
            for v in (r.get("attributes") or {}).get(a["name"], {}).get("values", []):
                if v.get("value"):
                    counts[v["value"]] = counts.get(v["value"], 0) + 1
        value_counts[a["name"]] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    dates = sorted(
        s["retrieved_at"][:10]
        for r in records
        for av in (r.get("attributes") or {}).values()
        for v in av.get("values", [])
        for s in v.get("sources") or []
        if s.get("retrieved_at")
    )
    return {
        "title": title,
        "version": version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "category": data.get("category"),
        "guidance": data.get("guidance"),
        "record_count": len(records),
        "sources_retrieved": {"earliest": dates[0], "latest": dates[-1]}
        if dates
        else None,
        "schema": [
            {
                k: a.get(k)
                for k in (
                    "name",
                    "description",
                    "is_closed_set",
                    "is_multi_valued",
                    "locked",
                    "canonical_values",
                    "canonical_value_descriptions",
                )
            }
            for a in schema
        ],
        "value_counts": value_counts,
        "exclusions": len(data.get("user_exclusions") or []),
        "history": data.get("history") or [],
        "notes": notes,
    }


def build_release_zip(
    session: Session, *, version: str, notes: str = "", **dashboard_kwargs
) -> bytes:
    """One zip for publication: data.json, data.csv, manifest.json and dashboard/.

    ``dashboard_kwargs`` are passed to :meth:`build_dashboard_zip`.
    """
    dash = zipfile.ZipFile(io.BytesIO(build_dashboard_zip(session, **dashboard_kwargs)))
    manifest = build_release_manifest(
        session,
        title=dashboard_kwargs.get("title", ""),
        version=version,
        notes=notes,
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.json", storage.get_dataset_bytes_json(session))
        zf.writestr("data.csv", get_release_csv_bytes(session))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for name in dash.namelist():
            zf.writestr(name, dash.read(name))
    buf.seek(0)
    return buf.read()


def build_dashboard_zip(
    session: Session,
    title: str,
    subtitle: str,
    dataset_label: str,
    primary_color: str,
    accent_color: str,
    logo_bytes: bytes | None = None,
    logo_filename: str | None = None,
    favicon_bytes: bytes | None = None,
    favicon_filename: str | None = None,
    views: list[str] | None = None,
    secondary_accent_color: str | None = None,
    colors: dict[str, str] | None = None,
    footer: str | None = None,
    labels: dict[str, str] | None = None,
    intro: str | None = None,
    intro_details: str | None = None,
) -> bytes:
    """Bundle dashboard.html + theme + data into a downloadable ZIP.

    ``views`` optionally restricts the dashboard's tab bar. Accepted
    values are ``"table"``, ``"cards"``, ``"network"``. ``None`` (the
    default) or an empty list keeps all three. The first listed view
    opens by default.

    ``labels`` overrides display text: ``entity`` (name column header,
    default "Entity"), ``sources`` (citation-count column, default
    "Sources") and ``sourcesHelp`` (tooltip explaining that count).
    ``intro`` is optional plain text shown above the summary tiles; keep it
    to one line. ``intro_details`` is optional longer plain text opened from
    an "About this map" header button as an overlay panel, so it takes no
    page space (blank lines start paragraphs).

    ``colors`` overrides any of the 12 theme color slots: primary,
    accent, accentSoft, accent2, warn, highlight, success, bg, surface,
    text, muted, border. Values not provided fall back to defaults
    derived from ``primary_color`` / ``accent_color`` /
    ``secondary_accent_color``.
    """
    dashboard_src = DASHBOARD_DIR / "dashboard.html"

    normalised_views: list[str] = []
    if views:
        allowed = {"table", "cards", "network"}
        for v in views:
            key = str(v).strip().lower()
            if key in allowed and key not in normalised_views:
                normalised_views.append(key)

    default_colors = {
        "primary": primary_color,
        "accent": accent_color,
        "accentSoft": _lighten(accent_color),
        "accent2": secondary_accent_color or accent_color,
        "warn": "#E85D4A",
        "highlight": "#F0A830",
        "success": "#4CAF50",
        "bg": "#F5F7FA",
        "surface": "#FFFFFF",
        "text": "#1A2A3A",
        "muted": "#6B7C8D",
        "border": "#DDE3EA",
    }
    if colors:
        for k, v in colors.items():
            if isinstance(v, str) and v:
                default_colors[k] = v

    theme = {
        "name": "custom",
        "title": title,
        "subtitle": subtitle,
        "datasetLabel": dataset_label,
        "logo": logo_filename or "",
        "logoAlt": subtitle,
        "favicon": favicon_filename or "",
        "footer": footer or "Built with Intelligence Toolkit.",
        "views": normalised_views,
        "labels": {
            k: v
            for k, v in (labels or {}).items()
            if k in ("entity", "sources", "sourcesHelp") and isinstance(v, str) and v
        },
        "intro": intro or "",
        "introDetails": intro_details or "",
        "colors": default_colors,
    }

    theme_js = "window.SCHEMIFY_THEME = " + json.dumps(theme, indent=2) + ";\n"
    data_js = (
        "const DASHBOARD_DATA = "
        + json.dumps(session.dataset_json or {}, ensure_ascii=False)
        + ";\n"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if dashboard_src.exists():
            zf.writestr("dashboard/dashboard.html", dashboard_src.read_bytes())
        zf.writestr("dashboard/theme.js", theme_js.encode())
        zf.writestr("dashboard/dashboard_data.js", data_js.encode())
        if logo_bytes and logo_filename:
            zf.writestr(f"dashboard/{logo_filename}", logo_bytes)
        if favicon_bytes and favicon_filename:
            zf.writestr(f"dashboard/{favicon_filename}", favicon_bytes)
    buf.seek(0)
    return buf.read()


def load_theme_bundle(files: list[tuple[str, bytes]]) -> dict:
    """Read dashboard branding from a release ZIP or from theme files.

    ``files`` holds ``(filename, bytes)`` pairs: a release/dashboard ZIP, or a
    ``theme.js`` / ``theme.json`` plus the logo and favicon it references.
    Returns ``{"theme": dict, "assets": {filename: bytes}}``; raises
    ``ValueError`` when no theme is found.
    """
    entries: dict[str, bytes] = {}
    for name, raw in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for member in zf.namelist():
                    if not member.endswith("/"):
                        entries[member] = zf.read(member)
        else:
            entries[name] = raw
    theme_name = next(
        (
            n
            for n in sorted(entries, key=len)
            if Path(n).name in ("theme.js", "theme.json")
        ),
        None,
    )
    if theme_name is None:
        msg = "No theme.js or theme.json found"
        raise ValueError(msg)
    text = entries[theme_name].decode("utf-8")
    if theme_name.endswith(".js"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            msg = f"Could not read the theme object in {theme_name}"
            raise ValueError(msg)
        text = text[start : end + 1]
    theme = json.loads(text)
    folder = str(Path(theme_name).parent).replace("\\", "/")
    assets: dict[str, bytes] = {}
    for key in ("logo", "favicon"):
        fname = theme.get(key) or ""
        if not fname:
            continue
        for candidate in (
            f"{folder}/{fname}" if folder not in ("", ".") else fname,
            fname,
        ):
            if candidate in entries:
                assets[fname] = entries[candidate]
                break
    return {"theme": theme, "assets": assets}


def theme_dashboard_kwargs(bundle: dict) -> dict:
    """Map a loaded theme bundle to :meth:`itk_entities.BuildEntityDataset.build_dashboard_zip` arguments."""
    theme = bundle.get("theme") or {}
    assets = bundle.get("assets") or {}
    colors = dict(theme.get("colors") or {})
    logo, favicon = theme.get("logo") or "", theme.get("favicon") or ""
    return {
        "title": theme.get("title", ""),
        "subtitle": theme.get("subtitle", ""),
        "dataset_label": theme.get("datasetLabel") or "entities",
        "primary_color": colors.get("primary", "#002B49"),
        "accent_color": colors.get("accent", "#0078D4"),
        "secondary_accent_color": colors.get("accent2"),
        "colors": colors,
        "logo_bytes": assets.get(logo),
        "logo_filename": logo if logo in assets else None,
        "favicon_bytes": assets.get(favicon),
        "favicon_filename": favicon if favicon in assets else None,
        "views": list(theme.get("views") or []),
        "footer": theme.get("footer"),
        "labels": dict(theme.get("labels") or {}),
        "intro": theme.get("intro") or None,
        "intro_details": theme.get("introDetails") or None,
    }


def _lighten(hex_color: str) -> str:
    """Return a light tint of a hex color (80% toward white)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = r + (255 - r) * 4 // 5
        g = g + (255 - g) * 4 // 5
        b = b + (255 - b) * 4 // 5
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:  # noqa: BLE001
        return "#E0F0FF"
