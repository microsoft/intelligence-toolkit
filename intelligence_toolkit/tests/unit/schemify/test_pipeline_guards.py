import json

import pytest

from intelligence_toolkit.schemify import Schemify
from intelligence_toolkit.schemify import scope_audit
from intelligence_toolkit.schemify.models import (
    AttributeValue,
    Citation,
    Record,
    RecordSet,
    SchemaAttribute,
    SchemifyConfig,
    SourcedValue,
)
from intelligence_toolkit.schemify.strategy_agentic import AgenticStrategy

FUNC = ["Risk Assessment & Monitoring", "Data Sharing & Analytics", "Other"]


class ForbiddenLLM:
    async def structured_completion(self, **kwargs):
        raise AssertionError("LLM must not be called for locked attributes")


class FixedLLM:
    def __init__(self, result):
        self.result = result

    async def structured_completion(self, **kwargs):
        return self.result


def av(*values):
    return AttributeValue(values=[
        SourcedValue(value=v, sources=[Citation(url=f"https://x/{i}", title="t")])
        for i, v in enumerate(values)
    ])


def values(record, attr):
    a = record.attributes.get(attr)
    return [sv.value for sv in a.values] if a else []


@pytest.fixture
def sch():
    return Schemify(SchemifyConfig(api_key="test", cache_enabled=False))


def make_rs(records, locked=True):
    return RecordSet(
        category="Tools",
        guidance="",
        records=records,
        schema_attributes=[
            SchemaAttribute(
                name="Functionality", is_closed_set=True, canonical_values=list(FUNC[:2]),
                canonical_value_descriptions={FUNC[0]: "scores risk"}, locked=locked,
            ),
        ],
    )


def test_finalize_drops_fallback_values_alongside_specific(sch):
    rs = make_rs([
        Record(label="A", attributes={"Functionality": av("Other", FUNC[0])}),
        Record(label="B", attributes={"Functionality": av("Other")}),
    ], locked=False)
    rs.schema_attributes.append(SchemaAttribute(
        name="Trafficking Type", is_closed_set=True,
        canonical_values=["Labor", "Not Applicable / General-Purpose"],
    ))
    rs.records[0].attributes["Trafficking Type"] = av("Not Applicable / General-Purpose", "Labor")

    stats = sch.resolution.finalize_normalization(rs)

    assert values(rs.records[0], "Functionality") == [FUNC[0]]
    assert values(rs.records[0], "Trafficking Type") == ["Labor"]
    assert values(rs.records[1], "Functionality") == ["Other"]
    assert stats["fallbacks_dropped"] == 2


def test_finalize_keeps_long_pinned_labels(sch):
    long_label = "Machine-learning inspection targeting model (Stanford Data Lab, Brazil)"
    rs = make_rs([Record(label=long_label, manual_canonical=True),
                  Record(label="Another quite long tool name here (with a long parenthetical qualifier)")])
    sch.resolution.finalize_normalization(rs)
    assert rs.records[0].label == long_label.upper()
    assert "(" not in rs.records[1].label


def test_finalize_snaps_locked_values_to_taxonomy(sch):
    rs = make_rs([
        Record(label="A", attributes={"Functionality": av("risk assessment & monitoring", "Invented Category")}),
    ])
    stats = sch.resolution.finalize_normalization(rs)
    assert values(rs.records[0], "Functionality") == [FUNC[0]]
    assert stats["off_taxonomy_dropped"] == 1


def test_locked_taxonomy_without_other_rejects_other(sch):
    from intelligence_toolkit.schemify.schemas import get_record_extraction_schema

    rs = make_rs([Record(label="A", attributes={"Functionality": av("Other")})])
    schema = get_record_extraction_schema(rs.schema_attributes, with_citations=True)
    props = schema["json_schema"]["schema"]["properties"]["records"]["items"]["properties"]
    assert "Other" not in props["Functionality"]["properties"]["value"]["enum"]
    sch.resolution.finalize_normalization(rs)
    assert values(rs.records[0], "Functionality") == []


def test_unlocked_taxonomy_still_offers_other():
    from intelligence_toolkit.schemify.schemas import get_record_extraction_schema

    attrs = make_rs([], locked=False).schema_attributes
    schema = get_record_extraction_schema(attrs, with_citations=False)
    props = schema["json_schema"]["schema"]["properties"]["records"]["items"]["properties"]
    assert props["Functionality"]["enum"][-1] == "Other"


def test_finalize_leaves_unlocked_values_alone(sch):
    rs = make_rs([Record(label="A", attributes={"Functionality": av("Invented Category")})], locked=False)
    sch.resolution.finalize_normalization(rs)
    assert values(rs.records[0], "Functionality") == ["Invented Category"]


def test_finalize_ignores_stale_additional_copies_of_taxonomy(sch):
    rec = Record(
        label="A",
        attributes={"Functionality": av(FUNC[0])},
        additional_attributes={"functionality": av("Case management (raw, pre-normalization)")},
    )
    rs = make_rs([rec], locked=False)
    stats = sch.resolution.finalize_normalization(rs)
    assert values(rec, "Functionality") == [FUNC[0]]
    assert "functionality" not in rec.additional_attributes
    assert stats["stale_taxonomy_copies_dropped"] == 1


def test_finalize_keeps_unpopulated_locked_attribute(sch):
    rs = make_rs([Record(label="A", attributes={"Name": av("x")})])
    rs.schema_attributes += [SchemaAttribute(name="Name"), SchemaAttribute(name="Empty Unlocked")]
    sch.resolution.finalize_normalization(rs)
    names = [a.name for a in rs.schema_attributes]
    assert "Functionality" in names
    assert "Empty Unlocked" not in names


def test_finalize_title_cases_snake_case_attribute_names(sch):
    rs = make_rs([Record(label="A", attributes={"tool_description": av("does things")})])
    rs.schema_attributes.append(SchemaAttribute(name="tool_description"))
    sch.resolution.finalize_normalization(rs)
    assert "Tool Description" in [a.name for a in rs.schema_attributes]
    assert values(rs.records[0], "Tool Description") == ["does things"]


def test_evolve_schema_never_demotes_locked(sch):
    records = [Record(label=f"R{i}", attributes={"Other Attr": av("v")}) for i in range(10)]
    records[0].attributes["Functionality"] = av(FUNC[0])
    rs = make_rs(records)
    sch.resolution.evolve_schema(rs)
    assert values(rs.records[0], "Functionality") == [FUNC[0]]


async def test_auto_normalize_and_enum_expansion_skip_locked(sch):
    rs = make_rs([Record(label=f"R{i}", attributes={"Functionality": av("Other", "Weird")}) for i in range(5)])
    sch.resolution.llm = ForbiddenLLM()
    await sch.resolution.auto_normalize(rs)
    assert await sch.resolution.expand_enum_values(rs) == {}
    assert rs.schema_attributes[0].canonical_values == FUNC[:2]


async def test_resolve_attribute_names_keeps_locked_name(sch):
    rs = make_rs([Record(label="A", attributes={"Functionality": av(FUNC[0]), "Feature": av("x")})])
    sch.resolution.llm = FixedLLM({"mappings": [
        {"original": "Functionality", "canonical": "Capability"},
        {"original": "Feature", "canonical": "Capability"},
    ]})
    await sch.resolution.resolve_attribute_names(rs)
    assert "Functionality" in rs.records[0].attributes
    assert "Capability" in rs.records[0].attributes


def test_agent_cannot_demote_rename_or_merge_locked_taxonomy(sch):
    rs = make_rs([Record(label="A", attributes={"Functionality": av(FUNC[0], FUNC[1], "Dashboards")})])
    agent = AgenticStrategy(sch.config, sch.llm, sch.extraction, sch.resolution)
    agent._apply_schema_changes(rs, [
        {"kind": "demote", "attribute": "Functionality"},
        {"kind": "rename", "attribute": "Functionality", "new_name": "Capability"},
    ])
    agent._apply_normalizations(rs, [
        {"attribute": "Functionality", "merge_values": [FUNC[1]], "canonical": FUNC[0]},
        {"attribute": "Functionality", "merge_values": ["Dashboards"], "canonical": FUNC[1]},
    ])
    assert [a.name for a in rs.schema_attributes] == ["Functionality"]
    assert sorted(values(rs.records[0], "Functionality")) == sorted([FUNC[0], FUNC[1], FUNC[1]])


def test_recordset_round_trip_preserves_taxonomy_and_history():
    rs = make_rs([])
    rs.history.append({"op": "recategorize", "value_translations": {"Functionality": {"a": "b"}}})
    restored = RecordSet.from_dict(json.loads(json.dumps(rs.to_dict())))
    attr = restored.schema_attributes[0]
    assert attr.locked
    assert attr.canonical_value_descriptions == {FUNC[0]: "scores risk"}
    assert restored.history == rs.history


def test_seed_state_restore_preserves_taxonomy(sch, tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(make_rs([]).to_dict()), encoding="utf-8")
    target = RecordSet(category="Tools", guidance="")
    agent = AgenticStrategy(sch.config, sch.llm, sch.extraction, sch.resolution)
    agent._restore_state(target, str(path))
    attr = target.schema_attributes[0]
    assert attr.locked and attr.canonical_values == FUNC[:2]
    assert attr.canonical_value_descriptions == {FUNC[0]: "scores risk"}


def test_reloaded_run_keeps_recorded_usage(tmp_path):
    from intelligence_toolkit.build_entity_dataset.api import BuildEntityDataset, UsageStats

    (tmp_path / "data.json").write_text(json.dumps(make_rs([]).to_dict()), encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps(
        {"total_cost_usd": 12.5, "total_tokens": 1000, "queries_run": 40}
    ), encoding="utf-8")
    api = BuildEntityDataset()
    api.load_saved_run(tmp_path / "data.json")
    api.usage = UsageStats(total_tokens=10, total_cost_usd=0.5, queries_run=2)
    api._save_run("Tools")
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["total_cost_usd"] == 13.0
    assert meta["queries_run"] == 42


async def test_seed_names_keep_aliases_and_carry_no_values(sch):
    target = make_rs([])
    agent = AgenticStrategy(sch.config, sch.llm, sch.extraction, sch.resolution)
    added = await agent._ingest_seed_records(target, [{"label": "TOOL X", "aliases": ["X App"]}])
    assert added == 1
    rec = target.records[0]
    assert rec.aliases == ["X App"]
    assert rec.attributes == {}


async def test_scope_audit_weak_relevance_is_review_only():
    rs = make_rs([Record(label="OKTA")])
    llm = FixedLLM({"results": [
        {"label": "OKTA", "in_scope": True, "weak_relevance": True,
         "entity_kind": "Product / Tool / Platform", "confidence": 0.9, "reason": "general IAM"},
    ]})
    results = await scope_audit.audit_scope(rs, llm)
    assert scope_audit.flagged_results(results) == []
    assert [r["label"] for r in scope_audit.weak_relevance_results(results)] == ["OKTA"]


def test_scope_exclusion_removes_only_exact_label():
    from intelligence_toolkit.build_entity_dataset.api import BuildEntityDataset

    api = BuildEntityDataset()
    api._schemify = Schemify(SchemifyConfig(api_key="test", cache_enabled=False))
    api._schemify.record_set = make_rs([
        Record(label="PERSONA"),
        Record(label="PERSONA CANDIDATE VERIFICATION", aliases=["Persona"]),
    ])
    removed = api.apply_scope_exclusions([{"label": "PERSONA", "entity_kind": "Other", "reason": "r"}])
    assert removed == 1
    assert [r.label for r in api._schemify.record_set.records] == ["PERSONA CANDIDATE VERIFICATION"]


class CapturingLLM(FixedLLM):
    async def structured_completion(self, **kwargs):
        self.kwargs = kwargs
        return self.result


async def test_relevance_audit_uses_cited_evidence_and_filters_not_explicit():
    rec_yes = Record(label="A", attributes={"Connection": AttributeValue(values=[SourcedValue(
        value="Built for forced-labour screening",
        sources=[Citation(url="https://x/1", title="Vendor page", snippet="detects forced labour")],
    )])})
    rs = make_rs([rec_yes, Record(label="B")])
    llm = CapturingLLM({"results": [
        {"label": "A", "explicit": True, "connection": "Forced labour", "confidence": 0.9, "reason": "r"},
        {"label": "B", "explicit": False, "connection": "None", "confidence": 0.9, "reason": "r"},
    ]})
    results = await scope_audit.audit_relevance(
        rs, llm, criterion="explicit forced labour link", evidence_attribute="Connection",
        connections=["Forced labour"],
    )
    sent = json.loads(llm.kwargs["variables"]["entities"])
    assert sent[0]["evidence_snippets"] == ["detects forced labour"]
    assert sent[1]["connection_statement"] == "(none found in sources)"
    assert [r["label"] for r in scope_audit.not_explicit_results(results)] == ["B"]


async def test_relevance_audit_errors_never_filter():
    rs = make_rs([Record(label="A")])
    results = await scope_audit.audit_relevance(
        rs, ForbiddenLLM(), criterion="c", evidence_attribute="Connection", connections=["X"],
    )
    assert results[0]["explicit"] is None
    assert scope_audit.not_explicit_results(results) == []


def test_citation_title_drops_pdf_table_fragments():
    raw = ("TRAFFICKING IN HUMAN BEINGS AND ICT May 2024 None | | | --- | --- | --- | --- "
           "TRAFFICKING IN HUMAN BEINGS AND ICT May 2024 | | |Co-funded")
    assert Citation(url="https://x", title=raw).title == "TRAFFICKING IN HUMAN BEINGS AND ICT May 2024"
    assert Citation(url="https://x", title="Normal | Page Title").title == "Normal | Page Title"
    assert Citation(url="https://x", title="word " * 100).title.endswith("\u2026")


async def test_relevance_audit_uses_extra_evidence_and_label_subset():
    rec = Record(label="A", attributes={"Trafficking Type": AttributeValue(values=[SourcedValue(
        value="Labor", sources=[Citation(url="https://x/2", title="t", snippet="used to flag forced labour risk")],
    )])})
    rs = make_rs([rec, Record(label="B")])
    llm = CapturingLLM({"results": []})
    await scope_audit.audit_relevance(
        rs, llm, criterion="c", evidence_attribute="Connection", connections=["X"],
        extra_evidence_attributes=["Trafficking Type"], labels=["a"],
    )
    sent = json.loads(llm.kwargs["variables"]["entities"])
    assert [b["label"] for b in sent] == ["A"]
    assert sent[0]["evidence_snippets"] == ["used to flag forced labour risk"]


async def test_search_focus_steers_search(sch):
    captured = {}

    class FakeSearch:
        async def search(self, query):
            captured["query"] = query
            return "text", []

    class FakeCache:
        def get(self, ns, **key):
            captured["key"] = key
            return None

        def set(self, value, ns, **key):
            pass

    async def no_extract(**kwargs):
        return None

    sch.extraction.search = FakeSearch()
    sch.extraction.cache = FakeCache()
    sch.extraction._extract_attributes_into_record = no_extract
    await sch.extraction.expand_record(
        Record(label="A"), make_rs([]), target_attributes=["Functionality"], search_focus="forced labour",
    )
    assert "forced labour" in captured["query"]
    assert captured["key"]["focus"] == "forced labour"


def test_import_and_lock_taxonomy_applies_immediately(tmp_path, monkeypatch):
    import intelligence_toolkit.build_entity_dataset.api as bed

    monkeypatch.setattr(bed, "_RUNS_DIR", tmp_path)
    rs = make_rs([Record(label="A", attributes={"Functionality": av(FUNC[0], FUNC[1])})], locked=False)
    api = bed.BuildEntityDataset()
    api.import_dataset(rs.to_dict(), api_key="test")
    assert api.can_continue_research()
    api.update_schema_attribute("Functionality", canonical_values=[FUNC[0]], locked=True)
    vals = [v["value"] for v in api.dataset_json["records"][0]["attributes"]["Functionality"]["values"]]
    assert vals == [FUNC[0]]


def test_load_dataset_cleans_citation_titles():
    from intelligence_toolkit.build_entity_dataset.api import BuildEntityDataset

    data = make_rs([Record(label="A", attributes={"Functionality": AttributeValue(values=[SourcedValue(
        value=FUNC[0], sources=[Citation(url="https://x", title="t")],
    )])})]).to_dict()
    data["records"][0]["attributes"]["Functionality"]["values"][0]["sources"][0]["title"] = (
        "REPORT May 2024 None | | | --- | --- |"
    )
    api = BuildEntityDataset()
    api.load_dataset(data)
    src = api.dataset_json["records"][0]["attributes"]["Functionality"]["values"][0]["sources"][0]
    assert src["title"] == "REPORT May 2024"


def _live_api(records):
    from intelligence_toolkit.build_entity_dataset.api import BuildEntityDataset

    api = BuildEntityDataset()
    api._schemify = Schemify(SchemifyConfig(api_key="test", cache_enabled=False))
    api._schemify.record_set = make_rs(records, locked=False)
    return api


def test_restore_records_lifts_exclusion_and_logs():
    api = _live_api([Record(label="A")])
    api._schemify.record_set.user_exclusions = [{"label": "B", "reason": "r"}]
    n = api.restore_records([Record(label="B", attributes={"Functionality": av(FUNC[0])}).to_dict()], "csam")
    rs = api._schemify.record_set
    assert n == 1 and [r.label for r in rs.records] == ["A", "B"]
    assert rs.user_exclusions == []
    assert rs.history[-1]["op"] == "restore_records"


def test_set_record_values_keeps_evidence_and_logs():
    api = _live_api([Record(label="A", attributes={"Functionality": av(FUNC[0])})])
    assert api.set_record_values("A", "Functionality", [FUNC[1]], note="per taxonomy")
    sv = api._schemify.record_set.records[0].attributes["Functionality"].values[0]
    assert sv.value == FUNC[1] and sv.sources
    assert api._schemify.record_set.history[-1]["old"] == [FUNC[0]]


def test_release_zip_contains_package(tmp_path):
    import io
    import zipfile

    api = _live_api([Record(label="A", aliases=["Alpha"], attributes={"Functionality": av(FUNC[0])})])
    api._dataset_json = api._build_dataset_json()
    z = zipfile.ZipFile(io.BytesIO(api.build_release_zip(
        version="1.0", title="T", subtitle="S", dataset_label="tools",
        primary_color="#000000", accent_color="#111111",
    )))
    names = set(z.namelist())
    assert {"data.json", "data.csv", "manifest.json", "dashboard/dashboard.html"} <= names
    manifest = json.loads(z.read("manifest.json"))
    assert manifest["record_count"] == 1 and manifest["value_counts"]["Functionality"] == {FUNC[0]: 1}
    csv = z.read("data.csv").decode()
    assert "Alpha" in csv and "https://x/0" in csv


def test_manual_canonical_label_survives_round_trip():
    from intelligence_toolkit.schemify.models import Record

    rec = Record(label="Tool (Org, Place)", alias_counts={"TOOL": 5}, manual_canonical=True)
    loaded = Record.from_dict(rec.to_dict())
    loaded._update_canonical_label()
    assert loaded.manual_canonical and loaded.label == "Tool (Org, Place)"
    assert "manual_canonical" not in Record(label="X").to_dict()


def test_mixed_timezone_sources_compare():
    from intelligence_toolkit.schemify.models import AttributeValue

    av_ = AttributeValue.from_dict({"value": "x", "values": [{"value": "x", "sources": [
        {"url": "https://a/1", "title": "", "retrieved_at": "2026-10-07T15:10:25"},
        {"url": "https://b/2", "title": "", "retrieved_at": "2026-10-20T13:04:08+00:00"},
    ]}]})
    ev = av_.to_dict()["evidence"]
    assert ev["source_count"] == 2 and ev["last_seen_at"].startswith("2026-10-20")


def test_dashboard_theme_labels_views_and_intro():
    import io
    import zipfile

    api = _live_api([Record(label="A", attributes={"Functionality": av(FUNC[0])})])
    api._dataset_json = api._build_dataset_json()
    z = zipfile.ZipFile(io.BytesIO(api.build_dashboard_zip(
        title="T", subtitle="S", dataset_label="tools",
        primary_color="#000000", accent_color="#111111",
        views=["Table", "cards", "bogus"],
        labels={"entity": "Tool name", "sources": "References", "sourcesHelp": "Pages cited.", "x": "ignored"},
        intro="About this map.", intro_details="More.",
    )))
    theme = json.loads(z.read("dashboard/theme.js").decode().split("=", 1)[1].strip().rstrip(";"))
    assert theme["views"] == ["table", "cards"]
    assert theme["labels"] == {"entity": "Tool name", "sources": "References", "sourcesHelp": "Pages cited."}
    assert theme["intro"] == "About this map." and theme["introDetails"] == "More."
    html = z.read("dashboard/dashboard.html").decode()
    assert "SCHEMIFY_LABELS" in html and 'id="intro-note"' in html and 'aria-controls="intro-details"' in html
    assert "schema_attributes.filter(a => state.facetAttrs.includes(a.name))" in html  # group-by: categorical only


async def test_scope_audit_failure_keeps_records():
    rs = make_rs([Record(label="A"), Record(label="B")])
    results = await scope_audit.audit_scope(rs, ForbiddenLLM())
    assert all(r["in_scope"] for r in results)
    assert scope_audit.flagged_results(results) == []


async def test_scope_audit_flags_confident_out_of_scope():
    rs = make_rs([Record(label="A"), Record(label="B")])
    llm = FixedLLM({"results": [
        {"label": "A", "in_scope": False, "entity_kind": "Report / Publication", "confidence": 0.9, "reason": "r"},
        {"label": "B", "in_scope": False, "entity_kind": "Other", "confidence": 0.4, "reason": "r"},
    ]})
    flagged = scope_audit.flagged_results(await scope_audit.audit_scope(rs, llm))
    assert [f["label"] for f in flagged] == ["A"]

@pytest.mark.asyncio
async def test_select_single_values_keeps_chosen_and_folds_same_text_sources():
    from intelligence_toolkit.schemify import value_consolidation as vc

    desc = AttributeValue(values=[
        SourcedValue(value="Scores supplier risk for forced labour.", sources=[Citation(url="https://a", title="a")]),
        SourcedValue(value="Scores Supplier Risk For Forced Labour", sources=[Citation(url="https://b", title="b")]),
        SourcedValue(value="An R package for fishing vessels.", sources=[Citation(url="https://c", title="c")]),
    ])
    rs = make_rs([Record(label="A", attributes={"Tool Description": desc}),
                  Record(label="B", attributes={"Tool Description": av("only one")})])
    rs.schema_attributes.append(SchemaAttribute(name="Tool Description", is_multi_valued=True))
    llm = FixedLLM({"results": [{"label": "A", "choice": 0, "reason": "fits"}]})
    choices = await vc.select_single_values(rs, llm, "Tool Description")
    assert [c["label"] for c in choices] == ["A"] and choices[0]["chosen"].startswith("Scores supplier")
    assert vc.apply_single_value_choices(rs, "Tool Description", choices) == 1
    kept = rs.records[0].attributes["Tool Description"].values
    assert [sv.value for sv in kept] == ["Scores supplier risk for forced labour."]
    assert {c.url for c in kept[0].sources} == {"https://a", "https://b"}
    assert rs.schema_attributes[-1].is_multi_valued is False


@pytest.mark.asyncio
async def test_select_single_values_invalid_choice_changes_nothing():
    from intelligence_toolkit.schemify import value_consolidation as vc

    rs = make_rs([Record(label="A", attributes={"Tool Description": av("x", "y")})])
    choices = await vc.select_single_values(rs, FixedLLM({"results": [{"label": "A", "choice": 7, "reason": ""}]}),
                                            "Tool Description")
    assert choices[0]["choice"] is None
    assert vc.apply_single_value_choices(rs, "Tool Description", choices) == 0
    assert values(rs.records[0], "Tool Description") == ["x", "y"]


def test_merge_value_variants_keeps_fullest_form_and_partners():
    from intelligence_toolkit.schemify import value_consolidation as vc

    rs = make_rs([
        Record(label="A", attributes={"Org": av(
            "National Center for Missing & Exploited Children",
            "National Center for Missing & Exploited Children (NCMEC)")}),
        Record(label="B", attributes={"Org": av("Recorded Future", "Recorded Future, Inc.")}),
        Record(label="C", attributes={"Org": av("Seafood Watch, Liberty Shared, and SFP",
                                                "Seafood Watch", "Liberty Shared", "SFP")}),
        Record(label="D", attributes={"Org": av("Thorn", "Tech Coalition")}),
    ])
    preview = vc.value_variant_changes(rs, "Org")
    assert [p["label"] for p in preview] == ["A", "B", "C"]
    assert vc.apply_value_variant_changes(rs, "Org") == 3
    assert values(rs.records[0], "Org") == ["National Center for Missing & Exploited Children (NCMEC)"]
    assert values(rs.records[1], "Org") == ["Recorded Future, Inc."]
    assert values(rs.records[2], "Org") == ["Seafood Watch", "Liberty Shared", "SFP"]
    assert values(rs.records[3], "Org") == ["Thorn", "Tech Coalition"]
    merged = rs.records[0].attributes["Org"].values[0]
    assert len(merged.sources) == 2
    assert len(rs.records[2].attributes["Org"].values[0].sources) == 2


def test_rename_attribute_value_updates_schema_and_records():
    api = _live_api([Record(label="A", attributes={"Functionality": av(FUNC[0])})])
    assert api.rename_attribute_value("Functionality", FUNC[0], "Risk", description="new") == 1
    schema = api._schemify.record_set.schema_attributes[0]
    assert "Risk" in schema.canonical_values and FUNC[0] not in schema.canonical_values
    assert schema.canonical_value_descriptions == {"Risk": "new"}
    assert values(api._schemify.record_set.records[0], "Functionality") == ["Risk"]

async def test_single_value_prompt_renders_entities():
    rec = Record(label="Tool", attributes={"Desc": av("first", "second")})
    llm = CapturingLLM({"results": [{"label": "Tool", "choice": 1, "reason": "r"}]})
    from intelligence_toolkit.schemify import value_consolidation as vc
    out = await vc.select_single_values(make_rs([rec]), llm, "Desc")
    rendered = llm.kwargs["prompt"].format(**llm.kwargs["variables"])
    assert '"second"' in rendered and "{" + "entities}" not in rendered
    assert out[0]["chosen"] == "second"


def test_merge_records_prefers_label_over_alias():
    api = _live_api([Record(label="THORN SAFER", aliases=["Safer"]), Record(label="SAFER")])
    assert api.merge_records("THORN SAFER", "SAFER") is True
    assert [r.label for r in api._schemify.record_set.records] == ["THORN SAFER"]


def test_remove_source_drops_claims_only_backed_by_it():
    rec = Record(label="A", attributes={
        "Tech": AttributeValue(values=[
            SourcedValue(value="Web", sources=[Citation(url="https://good/1", title="t"),
                                               Citation(url="https://bad/app", title="t")]),
            SourcedValue(value="Mobile", sources=[Citation(url="https://bad/app", title="t")]),
        ]),
        "Geo": AttributeValue(values=[SourcedValue(value="Europe", sources=[Citation(url="https://bad/app", title="t")])]),
    })
    api = _live_api([rec])
    assert api.remove_source("A", "bad/app") == 3
    r = api._schemify.record_set.records[0]
    assert values(r, "Tech") == ["Web"] and "Geo" not in r.attributes
    assert [c.url for c in r.attributes["Tech"].values[0].sources] == ["https://good/1"]


def test_citation_strips_utm_tracking_only():
    from intelligence_toolkit.schemify.models import Citation
    assert Citation(url="https://a.org/p?utm_source=openai", title="t").url == "https://a.org/p"
    assert Citation(url="https://a.org/p?id=7&utm_source=openai&UTM_medium=x#s", title="t").url == "https://a.org/p?id=7#s"
    assert Citation.from_dict({"url": "https://a.org/?q=a%20b", "title": "t"}).url == "https://a.org/?q=a%20b"
    from intelligence_toolkit.schemify.models import SourcedValue
    sv = SourcedValue.from_dict({"value": "v", "sources": [
        {"url": "https://a.org/p", "title": "t", "snippet": "s"},
        {"url": "https://a.org/p?utm_source=openai", "title": "t", "snippet": "s"}]})
    assert len(sv.sources) == 1
