import json
import sys
from types import SimpleNamespace

from intelligence_toolkit.build_entity_dataset.multilingual import make_query_translator
from intelligence_toolkit.schemify import audit
from intelligence_toolkit.schemify.cache import NoOpCache
from intelligence_toolkit.schemify.extraction import ExtractionEngine
from intelligence_toolkit.schemify.models import (
    AttributeValue,
    Citation,
    Record,
    RecordSet,
    SchemaAttribute,
    SchemifyConfig,
    SourcedValue,
)
from intelligence_toolkit.schemify.propose import _expand_remappings
from intelligence_toolkit.schemify.value_cleanup import clean_compound_values


def test_openai_audit_uses_responses_json_format(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="{}")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    assert audit._run_openai("message", "model", "key") == "{}"
    assert captured["text"] == {"format": {"type": "json_object"}}
    assert "response_format" not in captured


def test_audit_prompt_and_dashboard_defaults_are_bundled():
    from intelligence_toolkit.schemify.cli import DASHBOARD_DIR

    assert audit.load_prompt(DASHBOARD_DIR / "missing") == audit.DEFAULT_AUDIT_PROMPT
    assert (DASHBOARD_DIR / "dashboard.html").is_file()


def test_recategorization_updates_taxonomy_and_merges_sources():
    data = {
        "schema_attributes": [
            {"name": "Type", "is_multi_valued": True, "canonical_values": ["Old"]}
        ],
        "records": [{
            "label": "A",
            "attributes": {"Type": {"values": [
                {"value": "Old", "sources": [{"url": "https://one"}]},
                {"value": "Other", "sources": [{"url": "https://two"}]},
            ]}},
        }],
    }
    proposal = {
        "schema": [{"name": "Type", "canonical_values": ["New"], "is_multi_value": False}],
        "added_attributes": [{"name": "Status", "is_multi_value": True}],
        "record_remappings": [
            {"label": "A", "attribute": "Type", "old_value": "Old", "new_value": "New"},
            {"label": "A", "attribute": "Type", "old_value": "Other", "new_value": "New"},
        ],
    }

    result = audit.apply_recategorization(data, proposal)
    schema = {item["name"]: item for item in result["schema_attributes"]}
    values = result["records"][0]["attributes"]["Type"]["values"]
    assert schema["Type"]["is_multi_valued"] is False
    assert schema["Status"]["is_multi_valued"] is True
    assert schema["Type"]["canonical_values"] == ["New"]
    assert values == [{
        "value": "New",
        "sources": [{"url": "https://one"}, {"url": "https://two"}],
    }]


def test_unresolved_translation_does_not_mark_record_out_of_scope():
    data = {"records": [{
        "label": "A",
        "attributes": {"Type": {"values": [{"value": "Unknown"}]}},
    }]}

    remaps, out_of_scope, unresolved = _expand_remappings(
        data, {"value_translations": {"Type": {"Unknown": "UNRESOLVED"}}}, ["Type"]
    )
    assert remaps == []
    assert out_of_scope == []
    assert len(unresolved) == 1


async def test_extraction_schema_and_records_keep_aliases():
    from intelligence_toolkit.schemify.schemas import get_record_extraction_schema

    schema = get_record_extraction_schema(["Type"])
    record_schema = schema["json_schema"]["schema"]["properties"]["records"]["items"]
    assert record_schema["properties"]["aliases"]["items"] == {"type": "string"}
    assert "aliases" in record_schema["required"]

    class FakeLLM:
        _progress_context = None

        async def structured_completion(self, **kwargs):
            return {"records": [{"label": "TOOL", "aliases": ["Tool X"], "Type": "App"}]}

    engine = ExtractionEngine(SchemifyConfig(api_key="test"), FakeLLM(), NoOpCache())
    records = await engine._extract_records_from_text(
        "", [], RecordSet(category="Tools", guidance="", schema_attributes=[
            SchemaAttribute(name="Type")
        ])
    )
    assert records[0].aliases == ["Tool X"]


async def test_query_translation_ignores_unrequested_language():
    class FakeLLM:
        async def structured_completion(self, **kwargs):
            return {"translations": [
                {"code": "es", "query": "es query"},
                {"code": "fr", "query": "fr query"},
            ]}

    translator = make_query_translator(FakeLLM(), ["en", "es"])
    assert await translator("query") == [("query", "en"), ("es query", "es")]


async def test_compound_cleanup_updates_taxonomy_but_skips_locked():
    class FakeLLM:
        async def structured_completion(self, **kwargs):
            return {"results": [{
                "original": "Alpha / Beta",
                "cleaned": ["Alpha", "Beta"],
            }]}

    attr = SchemaAttribute(
        name="Type", is_closed_set=True, canonical_values=["Alpha / Beta"],
        canonical_value_descriptions={"Alpha / Beta": "A definition"},
    )
    record_set = RecordSet(
        category="Tools", guidance="",
        records=[Record(label="A", attributes={"Type": AttributeValue(values=[
            SourcedValue(value="Alpha / Beta", sources=[
                Citation(url="https://source", title="source")
            ])
        ])})],
        schema_attributes=[attr],
    )
    await clean_compound_values(record_set, [attr], FakeLLM())
    assert attr.canonical_values == ["Alpha", "Beta"]
    assert attr.canonical_value_descriptions == {
        "Alpha": "A definition", "Beta": "A definition"
    }

    locked = SchemaAttribute(
        name="Locked", is_closed_set=True, canonical_values=["A / B"], locked=True
    )

    class ForbiddenLLM:
        async def structured_completion(self, **kwargs):
            raise AssertionError("locked taxonomy must not be cleaned")

    await clean_compound_values(RecordSet(category="Tools", guidance=""), [locked], ForbiddenLLM())
    assert locked.canonical_values == ["A / B"]


def test_theme_schema_allows_generated_views():
    from pathlib import Path

    schema = json.loads(
        (Path(__file__).parents[3] / "schemify" / "dashboard" / "theme.schema.json")
        .read_text(encoding="utf-8")
    )
    assert schema["properties"]["views"]["items"]["enum"] == ["table", "cards", "network"]
