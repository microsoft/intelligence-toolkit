import io
import json
import zipfile

import pytest

from intelligence_toolkit.build_entity_dataset.api import (
    BuildEntityDataset,
    load_theme_bundle,
    theme_dashboard_kwargs,
)

THEME = {
    "name": "custom",
    "title": "Interactive Map",
    "subtitle": "Org",
    "datasetLabel": "tools",
    "logo": "logo.png",
    "logoAlt": "Org",
    "favicon": "favicon.svg",
    "footer": "Produced for Org.",
    "views": ["table", "cards"],
    "labels": {"entity": "Tool name", "sources": "References", "sourcesHelp": "Pages cited."},
    "intro": "",
    "introDetails": "About text.\n\nSecond paragraph.",
    "colors": {"primary": "#002B49", "accent": "#00A19A", "accentSoft": "#E0F5F4", "accent2": "#4FBFCE",
               "warn": "#E85D4A", "highlight": "#F0A830", "success": "#4CAF50", "bg": "#F5F7FA",
               "surface": "#FFFFFF", "text": "#1A2A3A", "muted": "#6B7C8D", "border": "#DDE3EA"},
}


def _theme_js(theme: dict) -> bytes:
    return ("window.SCHEMIFY_THEME = " + json.dumps(theme, indent=2) + ";\n").encode()


def _release_theme(blob: bytes) -> tuple[dict, dict]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        bundle = load_theme_bundle([("release.zip", blob)])
        return bundle["theme"], {n: zf.read(n) for n in zf.namelist()}


def test_theme_files_rebuild_identical_dashboard_theme():
    api = BuildEntityDataset()
    api.load_dataset({"category": "c", "guidance": "", "schema_attributes": [], "records": []})
    files = [("theme.js", _theme_js(THEME)), ("logo.png", b"PNG"), ("favicon.svg", b"<svg/>"), ("other.png", b"x")]
    blob = api.build_release_zip(version="1", **theme_dashboard_kwargs(load_theme_bundle(files)))
    theme, members = _release_theme(blob)
    assert theme == THEME
    assert members["dashboard/logo.png"] == b"PNG"
    assert members["dashboard/favicon.svg"] == b"<svg/>"

    # A previous release ZIP reproduces the same theme and assets again.
    again = api.build_release_zip(version="2", **theme_dashboard_kwargs(load_theme_bundle([("prev.zip", blob)])))
    assert _release_theme(again)[0] == THEME


def test_theme_json_without_assets_and_missing_theme():
    bundle = load_theme_bundle([("theme.json", json.dumps(THEME).encode())])
    kw = theme_dashboard_kwargs(bundle)
    assert kw["views"] == ["table", "cards"]
    assert kw["logo_bytes"] is None
    assert kw["intro_details"] == THEME["introDetails"]
    with pytest.raises(ValueError, match="No theme"):
        load_theme_bundle([("logo.png", b"PNG")])
