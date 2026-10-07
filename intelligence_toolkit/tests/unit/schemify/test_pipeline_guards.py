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
