# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Parity with the legacy ``intelligence_toolkit.build_entity_dataset`` class.

The same script of operations is run against the legacy class and against
``itk_entities.BuildEntityDataset``. After every step the return value (or the
error raised) and the full dataset must match. Only operations that need no
model call are scripted here.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from itk_entities import BuildEntityDataset
from itk_schemify.models import (
    AttributeValue,
    Citation,
    Record,
    RecordSet,
    SchemaAttribute,
    SourcedValue,
)

legacy_api = pytest.importorskip("intelligence_toolkit.build_entity_dataset.api")

_VOLATILE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?"  # ISO timestamps
    r"|\d{8}-\d{6}"  # run folder timestamps
)


def _value(text: str, *urls: str) -> SourcedValue:
    return SourcedValue(
        value=text, sources=[Citation(url=u, title="source") for u in urls]
    )


def _attribute(*values: SourcedValue) -> AttributeValue:
    return AttributeValue(values=list(values))


def sample_dataset() -> dict:
    """Five records with aliases, value variants, missing cells and unsourced values."""
    record_set = RecordSet(
        category="Case management tools",
        guidance="Tools used by NGOs",
        schema_attributes=[
            SchemaAttribute(name="Country", description="Where it is based"),
            SchemaAttribute(
                name="Type",
                description="Kind of tool",
                is_closed_set=True,
                canonical_values=["Hotline", "Database"],
            ),
            SchemaAttribute(name="Founded", description="Year founded"),
        ],
        records=[
            Record(
                label="ACME TRACKER",
                aliases=["Acme Track", "AT"],
                attributes={
                    "Country": _attribute(
                        _value("USA", "https://a.example/1"),
                        _value("United States", "https://b.example/2"),
                    ),
                    "Type": _attribute(_value("Hotline", "https://a.example/1")),
                    "Founded": _attribute(_value("2010", "https://a.example/1")),
                },
            ),
            Record(
                label="ACME TRACKER PRO",
                attributes={
                    "Country": _attribute(_value("USA", "https://a.example/pro")),
                    "Type": _attribute(_value("Database", "https://a.example/pro")),
                },
            ),
            Record(
                label="BETA SUITE",
                aliases=["Beta"],
                attributes={
                    "Country": _attribute(_value("Canada", "https://beta.example/")),
                    "Type": _attribute(_value("hotline", "https://beta.example/")),
                },
            ),
            Record(
                label="GAMMA TOOL",
                attributes={
                    "Country": _attribute(_value("U.S.A.")),
                    "Type": _attribute(_value("Other thing", "https://gamma.example/")),
                },
            ),
            Record(label="DELTA"),
        ],
    )
    return record_set.to_dict()


def _normalize(value: Any, root: Path) -> Any:
    """Reduce a result to plain data with timestamps and temp paths blanked out."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize(dataclasses.asdict(value), root)
    if isinstance(value, pd.DataFrame):
        return _normalize(value.to_csv(index=False), root)
    if isinstance(value, Path):
        return _normalize(str(value), root)
    if isinstance(value, bytes):
        if value[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(value)) as archive:
                return {
                    name: _normalize(archive.read(name), root)
                    for name in sorted(archive.namelist())
                }
        try:
            return _normalize(value.decode("utf-8"), root)
        except UnicodeDecodeError:
            return f"<{len(value)} bytes>"
    if isinstance(value, str):
        text = value.replace(str(root), "<root>").replace(
            str(root).replace("\\", "\\\\"), "<root>"
        )
        text = text.replace(root.as_posix(), "<root>").replace("\\", "/")
        return _VOLATILE.sub("<time>", text)
    if isinstance(value, dict):
        return {str(k): _normalize(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v, root) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(v, root) for v in value)
    return value


Step = tuple[str, Callable[[Any], Any]]


def _dashboard(api: Any) -> bytes:
    return api.build_dashboard_zip("Title", "Subtitle", "Dataset", "#112233", "#445566")


SCRIPT: list[Step] = [
    ("schema_attributes", lambda a: a.schema_attributes),
    ("current_dataframe", lambda a: a.current_dataframe()),
    ("count_unverified", lambda a: a.count_unverified()),
    ("count_incomplete", lambda a: a.count_incomplete()),
    ("can_continue_research", lambda a: a.can_continue_research()),
    ("is_read_only", lambda a: (a.is_read_only, a.read_only_reason, a.is_running)),
    ("list_alias_groups", lambda a: a.list_alias_groups()),
    (
        "list_alias_groups only aliases",
        lambda a: a.list_alias_groups(only_with_aliases=True),
    ),
    # values
    ("value_variant_changes", lambda a: a.value_variant_changes("Country")),
    ("merge_value_variants", lambda a: a.merge_value_variants("Country")),
    ("value_variant_changes Type", lambda a: a.value_variant_changes("Type")),
    ("merge_value_variants Type", lambda a: a.merge_value_variants("Type")),
    (
        "rename_attribute_value",
        lambda a: a.rename_attribute_value(
            "Type", "Database", "Data Platform", description="stores data"
        ),
    ),
    (
        "rename_attribute_value unknown",
        lambda a: a.rename_attribute_value("Type", "Nope", "Still Nope"),
    ),
    (
        "set_record_values",
        lambda a: a.set_record_values(
            "GAMMA TOOL",
            "Type",
            ["Hotline"],
            note="fix",
            source_url="https://c.example/3",
        ),
    ),
    (
        "set_record_values unknown record",
        lambda a: a.set_record_values("NOPE", "Type", ["Hotline"]),
    ),
    (
        "apply_single_value_choices",
        lambda a: a.apply_single_value_choices(
            "Country", [{"label": "ACME TRACKER", "value": "USA", "reason": "official"}]
        ),
    ),
    # curation
    (
        "rename_record_canonical",
        lambda a: a.rename_record_canonical("BETA SUITE", "BETA SUITE PLUS"),
    ),
    ("remove_alias", lambda a: a.remove_alias("ACME TRACKER", "AT")),
    ("remove_alias unknown", lambda a: a.remove_alias("ACME TRACKER", "nope")),
    ("split_alias", lambda a: a.split_alias("ACME TRACKER", "Acme Track")),
    ("do_not_merge_pairs", lambda a: a.do_not_merge_pairs),
    ("merge_records", lambda a: a.merge_records("ACME TRACKER", "ACME TRACKER PRO")),
    ("merge_records unknown", lambda a: a.merge_records("ACME TRACKER", "NOPE")),
    ("unpin_canonical", lambda a: a.unpin_canonical("BETA SUITE PLUS")),
    ("remove_source", lambda a: a.remove_source("ACME TRACKER", "b.example")),
    (
        "apply_alias_suggestion",
        lambda a: a.apply_alias_suggestion("GAMMA TOOL", ["DELTA", "NOPE"]),
    ),
    (
        "apply_alias_correction",
        lambda a: a.apply_alias_correction(
            "BETA SUITE PLUS",
            "Beta",
            belongs_to_label="GAMMA TOOL",
            rationale="wrong owner",
        ),
    ),
    ("list_alias_groups after curation", lambda a: a.list_alias_groups()),
    # schema
    (
        "add_schema_attribute",
        lambda a: a.add_schema_attribute(
            "Region",
            "Where it operates",
            is_closed_set=True,
            canonical_values=["North", "South"],
        ),
    ),
    (
        "add_schema_attribute duplicate",
        lambda a: a.add_schema_attribute("Region", "again"),
    ),
    (
        "update_schema_attribute",
        lambda a: a.update_schema_attribute(
            "Region",
            description="Operating region",
            canonical_values=["North", "South", "East"],
        ),
    ),
    (
        "update_schema_attribute unknown",
        lambda a: a.update_schema_attribute("Nope", description="x"),
    ),
    ("rename_attribute", lambda a: a.rename_attribute("Founded", "Year Founded")),
    ("rename_attribute unknown", lambda a: a.rename_attribute("Nope", "Still Nope")),
    ("remove_attribute", lambda a: a.remove_attribute("Region")),
    ("schema_attributes after edits", lambda a: a.schema_attributes),
    # exclusions
    (
        "add_label_exclusion",
        lambda a: a.add_label_exclusion("Omega", reason="not a tool"),
    ),
    (
        "add_label_exclusion again",
        lambda a: a.add_label_exclusion("omega", reason="still not"),
    ),
    (
        "add_attribute_exclusion equals",
        lambda a: a.add_attribute_exclusion(
            "Country", "equals", ["Canada"], reason="out of region"
        ),
    ),
    (
        "add_attribute_exclusion in",
        lambda a: a.add_attribute_exclusion("Country", "in", ["Mexico", "Peru"]),
    ),
    (
        "add_attribute_exclusion missing",
        lambda a: a.add_attribute_exclusion("Year Founded", "missing"),
    ),
    (
        "add_attribute_exclusion bad operator",
        lambda a: a.add_attribute_exclusion("Country", "sounds like", ["x"]),
    ),
    ("exclusions", lambda a: a.exclusions),
    ("remove_exclusion", lambda a: a.remove_exclusion("Omega")),
    ("remove_exclusion unknown", lambda a: a.remove_exclusion("Nope")),
    ("add_exclusion", lambda a: a.add_exclusion("GAMMA TOOL", reason="duplicate")),
    (
        "apply_scope_exclusions",
        lambda a: a.apply_scope_exclusions([
            {"label": "ACME TRACKER", "reason": "out of scope"}
        ]),
    ),
    ("exclusions after", lambda a: a.exclusions),
    # recategorization (no model calls)
    ("recategorization_gaps", lambda a: a.recategorization_gaps(["Type", "Country"])),
    ("recategorization_candidates", lambda a: a.recategorization_candidates(["Type"])),
    (
        "recategorization_candidates keywords",
        lambda a: a.recategorization_candidates(
            ["Type"], scope="keywords", candidate_keywords=["hotline"]
        ),
    ),
    # research inputs
    ("append_guidance", lambda a: a.append_guidance("Focus on NGOs in Europe")),
    ("append_guidance empty", lambda a: a.append_guidance("  ")),
    (
        "add_candidate_entities",
        lambda a: a.add_candidate_entities(["Epsilon", "Epsilon", " ", "Zeta"]),
    ),
    (
        "restore_records",
        lambda a: a.restore_records(
            [sample_dataset()["records"][3]], reason="restored"
        ),
    ),
    ("remove_record_by_label", lambda a: a.remove_record_by_label("Zeta")),
    ("remove_record_by_label unknown", lambda a: a.remove_record_by_label("NOPE")),
    ("refresh_progress", lambda a: (a.refresh_progress(), a.progress, a.usage)[1:]),
    # exports
    ("get_dataset_bytes_json", lambda a: json.loads(a.get_dataset_bytes_json())),
    ("get_dataset_bytes_csv", lambda a: a.get_dataset_bytes_csv()),
    ("get_release_csv_bytes", lambda a: a.get_release_csv_bytes()),
    (
        "build_release_manifest",
        lambda a: a.build_release_manifest(
            title="Release", version="1.0", notes="first"
        ),
    ),
    ("build_dashboard_zip", _dashboard),
    (
        "build_release_zip",
        lambda a: a.build_release_zip(
            version="1.0",
            notes="first",
            title="Title",
            subtitle="Subtitle",
            dataset_label="Dataset",
            primary_color="#112233",
            accent_color="#445566",
        ),
    ),
    # saved runs
    ("reload read-only", lambda a: a.load_saved_run(a.run_dir / "data.json")),
    (
        "read-only flags",
        lambda a: (a.is_read_only, a.read_only_reason, a.can_continue_research()),
    ),
    (
        "merge_records while read-only",
        lambda a: a.merge_records("BETA SUITE PLUS", "GAMMA TOOL"),
    ),
    ("add_label_exclusion while read-only", lambda a: a.add_label_exclusion("Omega")),
    ("current_dataframe read-only", lambda a: a.current_dataframe()),
    ("schema_attributes read-only", lambda a: a.schema_attributes),
    ("progress after reload", lambda a: a.progress),
    ("reset", lambda a: a.reset()),
    ("after reset", lambda a: (a.dataset_json, a.dataframe, a.run_dir, a.progress)),
]


def _run_script(api: Any, root: Path) -> list[tuple[str, Any, Any]]:
    trace = []
    for name, step in SCRIPT:
        try:
            outcome: Any = ("ok", _normalize(step(api), root))
        except Exception as error:  # noqa: BLE001 - an error is a result to compare too
            outcome = ("error", type(error).__name__, _normalize(str(error), root))
        trace.append((name, outcome, _normalize(api.dataset_json, root)))
    return trace


@pytest.fixture()
def traces(tmp_path, monkeypatch):
    monkeypatch.chdir(
        tmp_path
    )  # Schemify creates its cache folder in the working directory
    legacy_root = tmp_path / "legacy"
    new_root = tmp_path / "new"
    monkeypatch.setattr(legacy_api, "_RUNS_DIR", legacy_root)

    legacy = legacy_api.BuildEntityDataset()
    legacy.import_dataset(sample_dataset(), api_key="test")
    new = BuildEntityDataset(runs_dir=new_root)
    new.import_dataset(sample_dataset(), api_key="test")

    return _run_script(legacy, legacy_root), _run_script(new, new_root)


def test_every_scripted_step_matches_the_legacy_class(traces):
    legacy_trace, new_trace = traces

    for (name, legacy_outcome, legacy_data), (_, new_outcome, new_data) in zip(
        legacy_trace, new_trace, strict=True
    ):
        assert new_outcome == legacy_outcome, f"result of step '{name}' differs"
        assert new_data == legacy_data, f"dataset after step '{name}' differs"


def test_the_script_exercises_real_behaviour(traces):
    """Guard against a script that passes only because every step fails the same way."""
    _, new_trace = traces

    errors = [name for name, outcome, _ in new_trace if outcome[0] == "error"]
    assert len(errors) <= 3, errors
    datasets = [json.dumps(data, sort_keys=True) for _, _, data in new_trace]
    assert len(set(datasets)) >= 20


@pytest.mark.parametrize(
    ("filename", "raw"),
    [
        ("names.csv", b"name,notes\nAlpha,x\nBeta,y\nAlpha,z\n"),
        ("names.txt", b"Alpha\n\n  Beta  \nalpha\n"),
        ("names.json", json.dumps(["Alpha", "Beta", "Alpha"]).encode()),
        (
            "names.json",
            json.dumps({"records": [{"label": "Alpha"}, {"name": "Beta"}]}).encode(),
        ),
        ("names.xlsx", b"not really a spreadsheet"),
    ],
)
def test_file_parsers_match_the_legacy_class(filename, raw):
    def outcome(fn):
        try:
            return ("ok", fn(filename, copy.copy(raw)))
        except Exception as error:  # noqa: BLE001 - an error is a result to compare too
            return ("error", type(error).__name__, str(error))

    legacy = legacy_api.BuildEntityDataset
    assert outcome(BuildEntityDataset.parse_candidate_file) == outcome(
        legacy.parse_candidate_file
    )
    assert outcome(BuildEntityDataset.reference_labels_from_file) == outcome(
        legacy.reference_labels_from_file
    )


def test_saved_runs_are_listed_from_the_session_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(legacy_api, "_RUNS_DIR", tmp_path / "runs")
    legacy = legacy_api.BuildEntityDataset()
    legacy.import_dataset(sample_dataset())

    listed = BuildEntityDataset.list_saved_runs(tmp_path / "runs")

    assert listed == legacy_api.BuildEntityDataset.list_saved_runs()
    assert [run["entity_count"] for run in listed] == [5]
