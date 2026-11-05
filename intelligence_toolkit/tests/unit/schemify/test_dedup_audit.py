import asyncio
import csv
import json

from intelligence_toolkit.schemify import dedup_audit as da
from intelligence_toolkit.schemify.cli import main as cli_main
from intelligence_toolkit.schemify.models import (
    AttributeValue,
    Citation,
    Record,
    RecordSet,
    SchemaAttribute,
    SourcedValue,
    clean_alias,
)


def av(*values, url="https://example.org/a"):
    return AttributeValue(values=[
        SourcedValue(value=v, sources=[Citation(url=url, title="t")]) for v in values
    ])


def rec(label, org, desc, aliases=(), url="https://example.org/a"):
    r = Record(label=label, aliases=list(aliases))
    r.attributes["Operating Organization"] = av(org, url=url)
    r.attributes["Tool Description"] = av(desc, url=url)
    return r


def record_set(*records):
    return RecordSet(
        category="Anti-trafficking tools",
        guidance="",
        records=list(records),
        schema_attributes=[
            SchemaAttribute(name="Tool Description"),
            SchemaAttribute(name="Operating Organization"),
            SchemaAttribute(name="Organization Type"),
        ],
    )


class FixedLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def structured_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def unseen_set():
    return record_set(
        rec("UNSEEN APP", "Unseen UK", "Free app to report modern slavery concerns to the helpline.",
            url="https://www.unseenuk.org/download-the-app/"),
        rec("UNSEENUK MOBILE APP", "Unseen", "Mobile app to report modern slavery concerns to the helpline.",
            aliases=["Unseen App"], url="https://www.unseenuk.org/download-the-app/"),
        rec("BUSINESS PORTAL", "Unseen", "Portal helping businesses respond to modern slavery risks."),
        rec("STOP APP", "STOP THE TRAFFIK", "App for anonymous trafficking reports."),
        rec("ANTI-HUMAN TRAFFICKING PLATFORM", "Z-Axiss", "Analyst platform for detecting networks.",
            aliases=["STOP APP"]),
    )


def test_org_attribute_detection_skips_type_attributes():
    assert da.org_attribute_names(unseen_set())[0] == "Operating Organization"
    assert "Organization Type" not in da.org_attribute_names(unseen_set())


def test_normalize_org_strips_suffixes_country_tags_and_accents():
    assert da.normalize_org("Issara Institute, Inc.") == {"issara institute"}
    assert da.normalize_org("Unseen UK") == {"unseen"}
    assert da.normalize_org("Nasdaq, Inc.") == {"nasdaq"}
    assert da.normalize_org("Verité") == {"verite"}
    assert da.normalize_org("Academic research") == set()


def test_candidates_group_org_variants_and_shared_names():
    clusters = da.find_candidate_clusters(unseen_set())
    groups = [sorted(r.label for r in c.records) for c in clusters]
    assert ["BUSINESS PORTAL", "UNSEEN APP", "UNSEENUK MOBILE APP"] in groups
    assert ["ANTI-HUMAN TRAFFICKING PLATFORM", "STOP APP"] in groups


def test_audit_validates_llm_labels_and_aliases():
    llm = FixedLLM({
        "merge_sets": [
            {"canonical_label": "unseen app", "member_labels": ["UNSEEN APP", "UNSEENUK MOBILE APP", "GHOST"],
             "rationale": "same app"},
            {"canonical_label": "GHOST", "member_labels": ["GHOST", "OTHER"], "rationale": "x"},
        ],
        "alias_corrections": [
            {"record_label": "ANTI-HUMAN TRAFFICKING PLATFORM", "alias": "stop app",
             "belongs_to_label": "STOP APP", "rationale": "other org's app"},
            {"record_label": "STOP APP", "alias": "not-an-alias", "belongs_to_label": "", "rationale": ""},
        ],
        "confidence": 0.9,
        "reason": "ok",
    })
    rs = unseen_set()
    results = asyncio.run(da.audit_clusters(rs, llm, concurrency=1))
    assert llm.calls
    assert "Anti-trafficking tools" in llm.calls[0]["variables"]["category"]
    merges = {(m.canonical_label, tuple(m.member_labels)) for r in results for m in r.merge_sets}
    assert ("UNSEEN APP", ("UNSEEN APP", "UNSEENUK MOBILE APP")) in merges
    assert all("GHOST" not in m for _, m in merges)
    fixes = {(c.record_label, c.alias, c.belongs_to_label) for r in results for c in r.alias_corrections}
    assert fixes == {("ANTI-HUMAN TRAFFICKING PLATFORM", "STOP APP", "STOP APP")}


def test_apply_merge_set_keeps_all_names_and_logs_history():
    rs = unseen_set()
    rs.records[0].aliases = ["UNSEEN UK APP"]  # alias absent from alias_counts
    absorbed = da.apply_merge_set(rs, "UNSEEN APP", ["UNSEEN APP", "UNSEENUK MOBILE APP"], rationale="same app")
    merged = da.find_record(rs, "UNSEEN APP")
    assert absorbed == ["UNSEENUK MOBILE APP"]
    assert len(rs.records) == 4
    assert merged.label == "UNSEEN APP"
    assert merged.manual_canonical
    assert {"UNSEEN UK APP", "UNSEENUK MOBILE APP"} <= set(merged.aliases)
    assert "Unseen App" not in merged.aliases  # same name as the label, case aside
    assert len(merged.attributes["Tool Description"].values) == 2
    assert rs.history[-1]["op"] == "merge_duplicates"
    assert rs.history[-1]["absorbed"] == ["UNSEENUK MOBILE APP"]
    assert da.apply_merge_set(rs, "UNSEEN APP", ["UNSEEN APP", "UNSEENUK MOBILE APP"]) == []


def test_overlapping_merges_keep_the_first_pinned_label():
    rs = record_set(
        rec("BUSSOLA IA", "Stanford", "Case prioritisation"),
        rec("TIP TRIAGE TOOL", "Stanford", "Case prioritisation"),
        rec("BRAZIL TOOL", "Stanford", "Case prioritisation"),
    )
    da.apply_merge_set(rs, "BUSSOLA IA", ["BUSSOLA IA", "TIP TRIAGE TOOL"])
    da.apply_merge_set(rs, "TIP TRIAGE TOOL", ["TIP TRIAGE TOOL", "BRAZIL TOOL"])
    assert [r.label for r in rs.records] == ["BUSSOLA IA"]
    assert {"TIP TRIAGE TOOL", "BRAZIL TOOL"} <= set(rs.records[0].aliases)

    rs.records.append(rec("BUSSOLA PLATFORM", "Stanford", "Case prioritisation"))
    da.apply_merge_set(rs, "BUSSOLA PLATFORM", ["BUSSOLA PLATFORM", "BRAZIL TOOL"])
    assert [r.label for r in rs.records] == ["BUSSOLA IA"]


def test_apply_alias_correction_matches_truncated_alias():
    rs = record_set(rec("TRAFFIK ANALYSIS HUB", "STOP THE TRAFFIK", "Data hub", aliases=["STOP APP", "TA Hub"]))
    assert da.apply_alias_correction(rs, "TRAFFIK ANALYSIS HUB", "stop app", rationale="other tool")
    assert rs.records[0].aliases == ["TA Hub"]
    assert rs.history[-1]["op"] == "remove_alias"
    assert not da.apply_alias_correction(rs, "TRAFFIK ANALYSIS HUB", "TRAFFIK ANALYSIS HUB")

    rs = record_set(rec("X", "Org", "d", aliases=["Traffik Analysis Hub (TA Hub"]))
    assert da.apply_alias_correction(rs, "X", "Traffik Analysis Hub (TA Hub)")
    assert rs.records[0].aliases == []


def test_clean_alias_and_record_cleanup():
    assert clean_alias(" None ") == ""
    assert clean_alias("Inclusive Labor Monitoring (ILM") == "Inclusive Labor Monitoring (ILM)"
    assert clean_alias("ILM)") == "ILM"
    assert clean_alias("Laws On Crime) - Case Law Database") == "Laws On Crime - Case Law Database"
    r = Record(label="INCLUSIVE LABOR MONITORING", aliases=["None", "ILM", "Issara ILM (ILM", "Issara ILM (ILM)"])
    r.alias_counts = {"INCLUSIVE LABOR MONITORING": 2, "None": 1, "Issara ILM (ILM": 1}
    changes = r.clean_aliases()
    assert r.aliases == ["ILM", "Issara ILM (ILM)"]
    assert "None" not in r.alias_counts
    assert r.alias_counts["Issara ILM (ILM)"] == 1
    assert ("None", "") in changes
    r.add_alias("n/a")
    assert "n/a" not in r.aliases


def test_proposal_rows_respect_approval_column():
    result = da.DedupAuditResult(
        organization="Unseen", candidate_reason="r", confidence=0.9,
        merge_sets=[da.MergeSet("UNSEEN APP", ["UNSEEN APP", "UNSEENUK MOBILE APP"], "same")],
        alias_corrections=[da.AliasCorrection("ANTI-HUMAN TRAFFICKING PLATFORM", "STOP APP", "STOP APP", "x")],
    )
    low = da.DedupAuditResult(organization="o", confidence=0.2,
                              merge_sets=[da.MergeSet("STOP APP", ["STOP APP", "BUSINESS PORTAL"], "?")])
    rows = da.proposal_rows([result, low])
    assert rows[0]["kind"] == "remove_alias"
    approvals = {r["canonical_label"] or r["record_label"]: r["approve"] for r in rows}
    assert approvals == {"ANTI-HUMAN TRAFFICKING PLATFORM": "yes", "UNSEEN APP": "yes", "STOP APP": ""}
    rs = unseen_set()
    statuses = da.apply_proposal_rows(rs, rows)
    assert [s["status"] for s in statuses] == ["applied", "applied"]
    assert da.find_record(rs, "BUSINESS PORTAL") is not None
    assert "STOP APP" not in da.find_record(rs, "ANTI-HUMAN TRAFFICKING PLATFORM").aliases


def test_cli_dedup_apply_preserves_untouched_records(tmp_path):
    data = unseen_set().to_dict()
    data["records"][3]["attributes"]["Tool Description"]["values"][0]["sources"][0]["url"] = (
        "https://example.org/a?utm_source=openai"
    )
    data["records"][2]["aliases"] = ["None"]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    proposals = tmp_path / "proposals.csv"
    with proposals.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=da.PROPOSAL_FIELDS)
        w.writeheader()
        w.writerow({"proposal_id": 1, "kind": "merge", "approve": "yes", "canonical_label": "UNSEEN APP",
                    "member_labels": "UNSEEN APP\nUNSEENUK MOBILE APP", "rationale": "same"})
        w.writerow({"proposal_id": 2, "kind": "merge", "approve": "", "canonical_label": "STOP APP",
                    "member_labels": "STOP APP\nBUSINESS PORTAL"})
    out = tmp_path / "out.json"
    assert cli_main(["dedup-apply", str(src), "--proposals", str(proposals), "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    labels = [r["label"] for r in result["records"]]
    assert labels == ["UNSEEN APP", "BUSINESS PORTAL", "STOP APP", "ANTI-HUMAN TRAFFICKING PLATFORM"]
    assert result["records"][2] == data["records"][3]  # untouched, utm URL kept verbatim
    assert result["records"][1]["aliases"] == []
    assert [h["op"] for h in result["history"]] == ["clean_aliases", "merge_duplicates"]
    assert (tmp_path / "out.applied.csv").exists()


def test_cli_dedup_apply_keeps_values_when_pinned_member_survives(tmp_path):
    rs = record_set(
        rec("C TOOL", "Org", "c-desc", aliases=["A APP"]),
        rec("A APP", "Org", "a-desc"),
        rec("B APP", "Org", "b-desc"),
    )
    rs.records[2].manual_canonical = True
    src = tmp_path / "data.json"
    src.write_text(json.dumps(rs.to_dict()), encoding="utf-8")
    proposals = tmp_path / "p.json"
    proposals.write_text(json.dumps([{"proposal_id": 1, "kind": "merge", "approve": "yes",
                                      "canonical_label": "A APP", "member_labels": "A APP\nB APP"}]),
                         encoding="utf-8")
    out = tmp_path / "out.json"
    assert cli_main(["dedup-apply", str(src), "--proposals", str(proposals), "--out", str(out)]) == 0
    by_label = {r["label"]: r for r in json.loads(out.read_text(encoding="utf-8"))["records"]}
    descs = [v["value"] for v in by_label["B APP"]["attributes"]["Tool Description"]["values"]]
    assert sorted(descs) == ["a-desc", "b-desc"]
    assert [v["value"] for v in by_label["C TOOL"]["attributes"]["Tool Description"]["values"]] == ["c-desc"]

def test_merges_follow_history_not_misattached_aliases():
    rs = record_set(
        rec("H-2 VISA HOTLINE", "Mexico", "Mexican hotline", aliases=["National Hotline"]),
        rec("NATIONAL HOTLINE", "Polaris", "US hotline"),
        rec("HOTLINE WEB CHAT", "Polaris", "US hotline chat"),
        rec("HOTLINE DIGITAL CHANNELS", "Polaris", "US hotline channels"),
    )
    rs.records[2].manual_canonical = True
    da.apply_merge_set(rs, "NATIONAL HOTLINE", ["NATIONAL HOTLINE", "HOTLINE WEB CHAT"])
    da.apply_merge_set(rs, "NATIONAL HOTLINE", ["NATIONAL HOTLINE", "HOTLINE DIGITAL CHANNELS"])
    by_label = {r.label: r for r in rs.records}
    assert set(by_label) == {"H-2 VISA HOTLINE", "HOTLINE WEB CHAT"}
    assert by_label["H-2 VISA HOTLINE"].aliases == ["National Hotline"]
    assert {"NATIONAL HOTLINE", "HOTLINE DIGITAL CHANNELS"} <= set(by_label["HOTLINE WEB CHAT"].aliases)
    assert not da.apply_alias_correction(rs, "National Hotline", "H-2 VISA HOTLINE")