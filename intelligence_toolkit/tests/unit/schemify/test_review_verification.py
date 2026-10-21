"""Evidence provenance, coverage and non-mutation regressions."""

import copy
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest

from intelligence_toolkit.schemify.copilot import (
    CopilotCompletionClient,
    completion_from_events,
)
from intelligence_toolkit.schemify.public_sources import (
    PublicSourceRetriever,
    RetrievedSource,
    extract_text,
    public_addresses,
)
from intelligence_toolkit.schemify.review_verification import (
    bind_proposals,
    checks_csv,
    decision_checks,
    decisions_csv,
    evidence_bundle,
    fingerprint,
    load_proposals,
    parse_report,
    read_report,
    source_urls,
    validate_checks,
    verify_proposals,
)

TEXT = "Alpha is a deployed software platform operated by Example Org for identifying human trafficking risks."
URL = "https://example.org/alpha"


def dataset():
    return {
        "category": "Tools",
        "guidance": "Include documented anti-trafficking software",
        "records": [
            {
                "label": "ALPHA",
                "aliases": ["Alpha tool"],
                "attributes": {
                    "Tool Description": {
                        "value": "Historical claim",
                        "values": [
                            {
                                "value": "Historical claim",
                                "sources": [{"url": URL, "snippet": "Not fresh proof"}],
                            },
                        ],
                    },
                    "Operating Organization": {"value": "Example Org"},
                },
            },
            {
                "label": "BETA",
                "aliases": [],
                "attributes": {
                    "Description": {
                        "values": [{"sources": [{"url": "https://example.org/beta"}]}]
                    },
                },
            },
        ],
    }


def proposals():
    return load_proposals(
        json.dumps(
            {
                "items": [
                    {
                        "id": 0,
                        "current_label": "ALPHA",
                        "action": "repair_record",
                        "proposed_label": "Alpha",
                        "proposed_description": TEXT,
                        "alias_decisions": [
                            {"alias": "Alpha tool", "decision": "keep"}
                        ],
                        "checks_before_apply": ["Obtain human approval"],
                    }
                ]
            }
        ),
        "json",
    )


class Retriever:
    def __init__(self, fail=False):
        self.urls = []
        self.fail = fail

    async def retrieve(self, url):
        self.urls.append(url)
        return RetrievedSource(
            url=url,
            final_url=url,
            status="failed" if self.fail else "retrieved",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            text="" if self.fail else TEXT,
            error="HTTP 403" if self.fail else "",
            content_sha256=hashlib.sha256(TEXT.encode()).hexdigest(),
        )


class Client:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    async def structured_completion(self, prompt, response_format, variables):
        assert prompt == "{payload}"
        payload = json.loads(variables["payload"].split("\n\n", 1)[1])
        self.calls.append(payload)
        source = payload["fresh_documents"][0]["document_id"]
        return {
            "checks": [
                {
                    "id": check["id"],
                    "status": "inconclusive"
                    if check["kind"] == "prerequisite"
                    else "supported",
                    "reason": "Human approval remains outstanding"
                    if check["kind"] == "prerequisite"
                    else "Direct source statement",
                    "evidence": []
                    if check["kind"] == "prerequisite"
                    else [
                        {
                            "document_id": source,
                            "quote": "This quote was invented and is not in the source."
                            if self.invalid
                            else TEXT,
                        },
                    ],
                }
                for check in payload["checks"]
            ]
        }


def test_csv_and_json_import_equivalent():
    csv = (
        "record_id,current_label,recommended_action,proposed_label,proposed_description,"
        "aliases_to_keep,checks_before_apply\n"
        f"0,ALPHA,repair_record,Alpha,{TEXT},Alpha tool,Obtain human approval\n"
    )
    assert load_proposals(csv, "csv") == proposals()


@pytest.mark.parametrize(
    "change",
    [
        {"current_label": "Alpha tool"},
        {"record_id": -1},
        {"record_id": 9},
        {"alias_decisions": []},
        {"alias_decisions": [{"alias": "Invented", "decision": "keep"}]},
        {"related_record_ids": [0]},
        {"related_record_ids": [9]},
        {"action": "split_record", "split_entities": ["Alpha"]},
        {"dataset_fingerprint": "outdated"},
    ],
)
def test_binding_rejects_stale_or_incomplete_decisions(change):
    proposal = proposals()[0]
    proposal.update(change)
    with pytest.raises(ValueError, match=".+"):
        bind_proposals(dataset(), [proposal])


def test_every_alias_and_split_and_related_and_prerequisite_checked():
    proposal = proposals()[0]
    proposal.update(
        {
            "alias_decisions": [
                {"alias": f"alias{i}", "decision": "detach"} for i in range(15)
            ],
            "split_entities": ["Alpha", "Gamma"],
            "related_record_ids": [1],
        }
    )
    checks = decision_checks(proposal, dataset()["records"])
    assert len([c for c in checks if c["kind"] == "alias_detach"]) == 15
    assert len([c for c in checks if c["kind"] == "split_entity"]) == 2
    assert "related_1" in {c["id"] for c in checks}
    assert "prerequisite_0" in {c["id"] for c in checks}


def test_sources_include_both_sides_and_report_omissions():
    proposal = proposals()[0]
    proposal["related_record_ids"] = [1]
    proposal["references"] = ["https://example.org/extra"]
    selected, omitted = source_urls(proposal, dataset()["records"], 2)
    assert selected == [URL, "https://example.org/beta"]
    assert omitted == ["https://example.org/extra"]


def test_related_labels_must_match_canonical_records():
    proposal = proposals()[0]
    proposal["related_record_ids"] = [1]
    proposal["related_record_labels"] = {"1": "Wrong record"}
    with pytest.raises(ValueError, match="Stale related"):
        bind_proposals(dataset(), [proposal])


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "invented_quote", "unknown_source", "no_evidence"],
)
def test_invalid_decisive_proof_is_rejected(mutation):
    checks = [{"id": "identity", "claim": "Alpha", "kind": "identity"}]
    documents = {"page": {"status": "retrieved", "text": TEXT}}
    item = {
        "id": "identity",
        "status": "supported",
        "reason": "Proof",
        "evidence": [
            {"document_id": "page", "quote": TEXT},
        ],
    }
    value = {"checks": [item]}
    if mutation == "missing":
        value["checks"] = []
    elif mutation == "duplicate":
        value["checks"].append(copy.deepcopy(item))
    elif mutation == "invented_quote":
        item["evidence"][0]["quote"] = "An invented substantive quotation."
    elif mutation == "unknown_source":
        item["evidence"][0]["document_id"] = "historical"
    else:
        item["evidence"] = []
    with pytest.raises(ValueError, match=".+"):
        validate_checks(value, checks, documents)


@pytest.mark.parametrize("length", [19, 1201])
def test_quote_length_thresholds_are_enforced_even_for_exact_matches(length):
    quote = "x" * length
    with pytest.raises(ValueError, match="invalid fresh quotation"):
        validate_checks(
            {
                "checks": [
                    {
                        "id": "identity",
                        "status": "supported",
                        "reason": "Claim",
                        "evidence": [{"document_id": "page", "quote": quote}],
                    }
                ]
            },
            [{"id": "identity", "kind": "identity", "claim": "Alpha"}],
            {"page": {"status": "retrieved", "text": quote}},
        )


async def test_full_persistence_and_explicit_resume_without_mutation(tmp_path):
    data = dataset()
    before = copy.deepcopy(data)
    retriever, client = Retriever(), Client()
    report = await verify_proposals(
        data, proposals(), tmp_path, client, retriever=retriever
    )
    assert data == before
    assert report["complete"]
    assert report["summary"]["completed_records"] == 1
    assert report["results"][0]["status"] == "inconclusive"
    assert report["results"][0]["factual_status"] == "supported"
    assert report["summary"]["recommendations_applied"] == 0
    assert report["dataset_fingerprint"] == fingerprint(before)
    assert read_report(tmp_path) == report
    with pytest.raises(ValueError, match="supported ITK"):
        parse_report(json.dumps({**report, "version": 999}))
    assert retriever.urls == [URL]
    assert (tmp_path / "checks.csv").read_bytes().decode("utf-8-sig") == checks_csv(
        report
    )
    assert (tmp_path / "decisions.csv").read_bytes().decode(
        "utf-8-sig"
    ) == decisions_csv(report)
    assert len(list(csv.DictReader(decisions_csv(report).splitlines()))) == 1
    document = next(iter(report["documents"].values()))
    assert document["retrieved_at"]
    assert document["content_sha256"]
    assert "text" not in document
    with ZipFile(BytesIO(evidence_bundle(tmp_path))) as archive:
        assert "report.json" in archive.namelist()
        assert "records/0.json" in archive.namelist()
        assert any(name.startswith("documents/") for name in archive.namelist())
    resumed_client, resumed_retriever = Client(), Retriever()
    resumed = await verify_proposals(
        data,
        proposals(),
        tmp_path,
        resumed_client,
        retriever=resumed_retriever,
        resume=True,
    )
    assert not resumed_client.calls
    assert not resumed_retriever.urls
    assert resumed["documents"] == report["documents"]
    assert resumed["summary"]["reused_sources"] == 1
    assert len(resumed["usage_sessions"]) == 2
    assert resumed["usage_sessions"][0] == report["usage_sessions"][0]
    with pytest.raises(ValueError, match="empty output"):
        await verify_proposals(
            data, proposals(), tmp_path, Client(), retriever=Retriever()
        )
    data["records"][0]["attributes"]["Operating Organization"]["value"] = "Changed"
    with pytest.raises(ValueError, match="changed"):
        await verify_proposals(
            data, proposals(), tmp_path, Client(), retriever=Retriever(), resume=True
        )


async def test_exported_capture_can_resume_without_paying_to_recheck(tmp_path):
    data = dataset()
    bound = proposals()
    bound[0]["dataset_fingerprint"] = fingerprint(data)
    original = tmp_path / "original"
    await verify_proposals(data, bound, original, Client(), retriever=Retriever())
    restored = tmp_path / "restored"
    with ZipFile(BytesIO(evidence_bundle(original))) as archive:
        archive.extractall(restored)
    imported = load_proposals(
        (restored / "proposals.json").read_text(encoding="utf-8"), "json"
    )
    assert imported == bound
    client, retriever = Client(), Retriever()
    report = await verify_proposals(
        data, imported, restored, client, retriever=retriever, resume=True
    )
    assert report["complete"]
    assert not client.calls
    assert not retriever.urls


async def test_failed_retrieval_is_explicit_and_cannot_support_anything(tmp_path):
    client = Client()
    report = await verify_proposals(
        dataset(), proposals(), tmp_path, client, retriever=Retriever(fail=True)
    )
    assert report["summary"]["processing_errors"] == 1
    assert report["summary"]["failed_sources"] == 1
    assert not client.calls
    assert all(c["status"] == "inconclusive" for c in report["results"][0]["checks"])
    assert next(iter(report["documents"].values()))["error"] == "HTTP 403"


async def test_retry_failed_sources_preserves_history_and_reassesses(tmp_path):
    data = dataset()
    data["records"][0]["attributes"]["Other"] = {
        "values": [{"sources": [{"url": "https://example.org/secondary"}]}],
    }

    class PartialRetriever(Retriever):
        async def retrieve(self, url):
            self.fail = url.endswith("secondary")
            return await super().retrieve(url)

    original = await verify_proposals(
        data, proposals(), tmp_path, Client(), retriever=PartialRetriever()
    )
    assert original["summary"]["processing_errors"] == 0
    recovered_retriever, recovered_client = Retriever(), Client()
    recovered = await verify_proposals(
        data,
        proposals(),
        tmp_path,
        recovered_client,
        retriever=recovered_retriever,
        resume=True,
        retry_failed_sources=True,
    )
    assert recovered_retriever.urls == ["https://example.org/secondary"]
    assert len(recovered_client.calls) == 1
    assert len(recovered_client.calls[0]["fresh_documents"]) == 2
    assert recovered["summary"]["recovered_sources"] == 1
    assert recovered["summary"]["retried_sources"] == 1
    assert recovered["summary"]["reused_sources"] == 1
    documents = {d["url"]: d for d in recovered["documents"].values()}
    original_documents = {d["url"]: d for d in original["documents"].values()}
    assert documents[URL]["retrieved_at"] == original_documents[URL]["retrieved_at"]
    secondary = documents["https://example.org/secondary"]
    assert secondary["attempt_history"][0]["status"] == "failed"
    assert secondary["attempt_history"][0]["error"] == "HTTP 403"
    assert secondary["retrieved_at"] >= secondary["attempt_history"][0]["retrieved_at"]


async def test_retry_requires_explicit_resume(tmp_path):
    with pytest.raises(ValueError, match="explicit resume"):
        await verify_proposals(
            dataset(), proposals(), tmp_path, Client(), retry_failed_sources=True
        )


async def test_invalid_response_retried_then_recorded_as_error(tmp_path):
    client = Client(invalid=True)
    report = await verify_proposals(
        dataset(), proposals(), tmp_path, client, retriever=Retriever()
    )
    assert len(client.calls) == 2
    assert report["summary"]["processing_errors"] == 1
    assert len(report["results"][0]["attempt_errors"]) == 2
    assert all(c["status"] == "inconclusive" for c in report["results"][0]["checks"])


async def test_contradictions_are_not_forced_to_pass(tmp_path):
    class ContradictingClient(Client):
        async def structured_completion(self, prompt, response_format, variables):
            reply = await super().structured_completion(
                prompt, response_format, variables
            )
            reply["checks"][0]["status"] = "contradicted"
            reply["checks"][0]["reason"] = "Source contradicts the proposed identity"
            return reply

    report = await verify_proposals(
        dataset(), proposals(), tmp_path, ContradictingClient(), retriever=Retriever()
    )
    assert report["results"][0]["status"] == "contradicted"
    assert report["summary"]["record_statuses"] == {"contradicted": 1}


async def test_pause_preserves_a_resumable_partial_capture(tmp_path):
    retriever = Retriever()
    paused = await verify_proposals(
        dataset(),
        proposals(),
        tmp_path,
        Client(),
        retriever=retriever,
        should_stop=lambda: True,
    )
    assert not paused["complete"]
    assert paused["stopped"]
    assert paused["summary"]["pending_records"] == 1
    assert not retriever.urls
    resumed = await verify_proposals(
        dataset(),
        proposals(),
        tmp_path,
        Client(),
        retriever=retriever,
        resume=True,
    )
    assert resumed["complete"]
    assert resumed["summary"]["pending_records"] == 0


def wait_for_verification(api):
    deadline = time.monotonic() + 10
    while api.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not api.is_running


def prepare_api(monkeypatch, tmp_path):
    from intelligence_toolkit.build_entity_dataset import api as api_module
    from intelligence_toolkit.schemify import review_verification as review

    monkeypatch.setattr(api_module, "_RUNS_DIR", tmp_path)
    monkeypatch.setattr(review, "make_client", lambda *args, **kwargs: Client())
    monkeypatch.setattr(review, "PublicSourceRetriever", Retriever)
    api = api_module.BuildEntityDataset()
    api.load_dataset(dataset())
    return api


def test_background_api_works_read_only_and_does_not_normalize_or_save_dataset(
    monkeypatch, tmp_path
):
    api = prepare_api(monkeypatch, tmp_path)
    before = copy.deepcopy(api.dataset_json)
    previous_progress = api.progress
    with patch.object(
        api, "_snapshot_partial", side_effect=AssertionError("Must not normalize")
    ):
        api.start_review_verification(proposals(), backend="copilot")
        wait_for_verification(api)
    assert api.dataset_json == before
    assert api.progress is previous_progress
    assert api.review_verification_progress.is_complete
    assert not api.review_verification_progress.error
    saved = api.list_review_verifications()
    assert len(saved) == 1
    assert not (tmp_path / "data.json").exists()
    api.open_review_verification(saved[0]["directory"])
    assert api.review_verification_report["summary"]["completed_records"] == 1
    assert api.review_verification_bundle()
    api.resume_review_verification(saved[0]["directory"])
    wait_for_verification(api)
    assert api.review_verification_report["summary"]["reused_sources"] == 1
    assert api.dataset_json == before


def test_api_resume_preserves_recorded_openai_model(monkeypatch, tmp_path):
    from intelligence_toolkit.schemify import review_verification as review

    api = prepare_api(monkeypatch, tmp_path)
    seen = []

    class ConfiguredClient(Client):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(completion_model="recorded-model")

    def make_client(backend, **kwargs):
        seen.append(kwargs)
        return ConfiguredClient()

    monkeypatch.setattr(review, "make_client", make_client)
    api.start_review_verification(proposals(), backend="openai", model="recorded-model")
    wait_for_verification(api)
    directory = api.list_review_verifications()[0]["directory"]
    api.resume_review_verification(directory)
    wait_for_verification(api)
    assert seen[-1]["model"] == "recorded-model"
    assert api.review_verification_report["complete"]
    assert not api.review_verification_progress.error


def test_api_can_retry_source_access_holds_without_mutating_dataset(
    monkeypatch, tmp_path
):
    from intelligence_toolkit.schemify import review_verification as review

    api = prepare_api(monkeypatch, tmp_path)
    before = fingerprint(api.dataset_json)
    monkeypatch.setattr(review, "PublicSourceRetriever", lambda: Retriever(fail=True))
    api.start_review_verification(proposals(), backend="copilot")
    wait_for_verification(api)
    assert api.review_verification_report["summary"]["processing_errors"] == 1
    directory = api.list_review_verifications()[0]["directory"]
    monkeypatch.setattr(review, "PublicSourceRetriever", Retriever)
    api.resume_review_verification(directory, retry_failed_sources=True)
    wait_for_verification(api)
    assert api.review_verification_report["summary"]["recovered_sources"] == 1
    assert api.review_verification_report["summary"]["processing_errors"] == 0
    assert not api.review_verification_progress.error
    assert fingerprint(api.dataset_json) == before


def test_cli_uses_same_verifier_and_preserves_exact_dataset_bytes(
    monkeypatch, tmp_path
):
    from intelligence_toolkit.schemify import cli
    from intelligence_toolkit.schemify import review_verification as review

    monkeypatch.setattr(review, "make_client", lambda *args, **kwargs: Client())
    monkeypatch.setattr(review, "PublicSourceRetriever", Retriever)
    data_path = tmp_path / "data.json"
    proposal_path = tmp_path / "proposals.json"
    data_path.write_text(json.dumps(dataset()), encoding="utf-8")
    proposal_path.write_text(json.dumps({"items": proposals()}), encoding="utf-8")
    before = data_path.read_bytes()
    argv = [
        "verify-review",
        str(data_path),
        "--proposals",
        str(proposal_path),
        "--out",
        str(tmp_path / "capture"),
        "--backend",
        "copilot",
    ]
    assert cli.main(argv) == 0
    assert cli.main([*argv, "--resume"]) == 0
    assert cli.main([*argv, "--resume", "--retry-failed-sources"]) == 0
    assert data_path.read_bytes() == before
    report = json.loads(
        (tmp_path / "capture" / "report.json").read_text(encoding="utf-8")
    )
    assert report["complete"]
    assert report["application_status"] == "not_applied"


def test_review_interface_starts_and_reopens_read_only_capture(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    api = prepare_api(monkeypatch, tmp_path)
    before = fingerprint(api.dataset_json)

    class Upload:
        name = "proposals.json"

        def getvalue(self):
            return json.dumps({"items": proposals()}).encode()

    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *args, **kwargs: Upload()
        if kwargs["key"] == "bed_review_upload"
        else None,
    )
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from app.workflows.build_entity_dataset.review_verification import render_review_verification\n"
        "render_review_verification(st.session_state['verification_api'], api_key='', model='unused')\n"
    )
    app.session_state["verification_api"] = api
    # First run imports the app package cold, which is slow on CI runners.
    app.run(timeout=120)
    assert not app.exception
    assert app.button(key="bed_review_start").disabled
    app.selectbox(key="bed_review_backend").select_index(1)
    app.checkbox(key="bed_review_authorized").check()
    app.run(timeout=60)
    app.button(key="bed_review_start").click().run(timeout=60)
    wait_for_verification(api)
    app.run(timeout=60)
    assert not app.exception
    assert not app.error
    assert api.review_verification_report["complete"]
    assert before == fingerprint(api.dataset_json)
    app.button(key="bed_review_open").click().run(timeout=60)
    assert not app.exception
    current = api.dataset_json
    assert current is not None
    current["records"][0]["label"] = "Changed"
    app.run(timeout=60)
    assert any("STALE REPORT" in warning.value for warning in app.warning)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://name:secret@example.org/",
        "https://example.org:8080/",
        "https://127.0.0.1/",
        "http://169.254.169.254/",
        "http://224.0.0.1/",
    ],
)
def test_nonpublic_urls_rejected(url):
    with pytest.raises(ValueError, match=".+"):
        public_addresses(url)


def test_mixed_public_private_dns_rejected():
    addresses = [
        (2, 1, 6, "", ("8.8.8.8", 443)),
        (2, 1, 6, "", ("10.0.0.1", 443)),
    ]
    with (
        patch("socket.getaddrinfo", return_value=addresses),
        pytest.raises(ValueError, match="public"),
    ):
        public_addresses("https://example.org")


def test_retrieval_failure_has_timestamp():
    result = PublicSourceRetriever().retrieve_sync("file:///etc/passwd")
    assert result.status == "failed"
    assert result.retrieved_at
    assert result.error


def test_connection_is_pinned_and_private_redirect_is_rejected():
    response = SimpleNamespace(status=302, headers=Message())
    response.headers["Location"] = "http://127.0.0.1/private"

    def addresses(host, *args, **kwargs):
        address = "8.8.8.8" if host == "example.org" else "127.0.0.1"
        return [(2, 1, 6, "", (address, 443))]

    with (
        patch("socket.getaddrinfo", side_effect=addresses),
        patch("socket.create_connection") as connect,
        patch("ssl.create_default_context"),
        patch("http.client.HTTPSConnection") as connection,
    ):
        connection.return_value.getresponse.return_value = response
        result = PublicSourceRetriever().retrieve_sync("https://example.org/start")
        connect.assert_called_once_with(("8.8.8.8", 443), timeout=25.0)
    assert result.status == "failed"
    assert "public" in result.error
    assert result.final_url == "http://127.0.0.1/private"


def test_https_request_uses_default_https_host_without_port_suffix():
    headers = Message()
    headers["Content-Type"] = "text/html"
    response = SimpleNamespace(
        status=200,
        headers=headers,
        read=lambda limit: (TEXT + TEXT).encode(),
    )
    raw_socket = MagicMock()
    with (
        patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]),
        patch("socket.create_connection", return_value=raw_socket),
        patch("ssl.create_default_context") as context,
        patch("http.client.HTTPConnection.getresponse", return_value=response),
    ):
        context.return_value.wrap_socket.return_value = raw_socket
        result = PublicSourceRetriever().retrieve_sync("https://example.org/alpha")
    assert result.status == "retrieved"
    request = b"".join(call.args[0] for call in raw_socket.sendall.call_args_list)
    assert b"Host: example.org\r\n" in request
    assert b"Host: example.org:443" not in request


def test_text_snapshot_truncation_threshold_is_exact():
    text, truncated = extract_text(b"x" * 250_001, "text/plain")
    assert len(text) == 250_000
    assert truncated


def test_pdf_page_limit_is_exact(monkeypatch):
    from intelligence_toolkit.schemify import public_sources

    calls = []

    class Page:
        def extract_text(self):
            calls.append(1)
            return TEXT

    monkeypatch.setattr(
        public_sources,
        "PdfReader",
        lambda raw: SimpleNamespace(pages=[Page() for _ in range(201)]),
    )
    text, truncated = extract_text(b"mock PDF", "application/pdf")
    assert len(calls) == 200
    assert text.count(TEXT) == 200
    assert truncated


def test_html_extraction_ignores_scripts_and_exposes_browser_only_pages():
    text, truncated = extract_text(
        ("<script>Untrusted instructions</script><p>" + TEXT + TEXT + "</p>").encode(),
        "text/html",
    )
    assert "Untrusted" not in text
    assert TEXT in text
    assert not truncated
    with pytest.raises(ValueError, match="browser"):
        extract_text(b"<div id='app'></div>", "text/html")


def events(content='{"items": []}', **data):
    return "\n".join(
        json.dumps(event)
        for event in [
            {
                "type": "assistant.message",
                "data": {"phase": "analysis", "content": "not final"},
            },
            {
                "type": "assistant.message",
                "data": {
                    "phase": "final_answer",
                    "content": content,
                    "model": "default",
                    **data,
                },
            },
            {"type": "result", "exitCode": 0, "usage": {"premiumRequests": 1}},
        ]
    )


def test_copilot_events_preserve_unwrapped_content_and_usage():
    content = json.dumps({"items": [{"description": "x" * 500}]})
    value, metadata = completion_from_events(events(content))
    assert value == content
    assert metadata["model"] == "default"
    assert metadata["usage"]["premiumRequests"] == 1


@pytest.mark.parametrize(
    "output",
    [
        events(toolRequests=[{"name": "shell"}]),
        '{"type":"session.error","data":{"message":"authentication required"}}',
        events().replace('"exitCode": 0', '"exitCode": 1'),
        '{"type":"assistant.message","data":{"phase":"final_answer","content":"{}"}}',
    ],
)
def test_copilot_failures_are_not_success_shaped(output):
    with pytest.raises((ValueError, RuntimeError)):
        completion_from_events(output)


async def test_copilot_uses_stdin_and_no_tools():
    with patch("shutil.which", return_value="copilot"), patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = events()
        client = CopilotCompletionClient()
        response = await client.structured_completion(
            "{payload}",
            {"json_schema": {"schema": {"type": "object"}}},
            {"payload": "Entire evidence payload"},
        )
        assert response == {"items": []}
        args, kwargs = run.call_args
        assert "--available-tools=" in args[0]
        assert "--disable-builtin-mcps" in args[0]
        assert "-p" not in args[0]
        assert "--model" not in args[0]
        assert kwargs["input"].startswith("Entire evidence payload")
