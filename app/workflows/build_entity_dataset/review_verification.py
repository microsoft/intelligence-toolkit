"""Review-only evidence verification controls, independent of extraction state."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from intelligence_toolkit.schemify.review_verification import (
    STATUSES,
    checks_csv,
    decisions_csv,
    load_proposals,
    parse_report,
)

if TYPE_CHECKING:
    from intelligence_toolkit.build_entity_dataset.api import BuildEntityDataset


def _render_report(report: dict) -> None:
    st.json(report["summary"], expanded=False)
    errors = report["summary"]["processing_errors"]
    if errors:
        st.warning(
            f"{errors} records have retrieval/processing errors and remain on hold. "
            "Inspect the Error column and per-record source failures; these records are not validated."
        )
    st.caption(
        "Record status includes outstanding prerequisites. Factual status excludes "
        "those workflow/human steps. Supported means evidence support, not tool "
        "quality, effectiveness or approval to apply. Contradicted means a check "
        "conflicts with fresh evidence; it can also resolve an open alias/scope "
        "question. Read the reasons rather than treating it as rejection of the whole proposal."
    )
    statuses = st.multiselect(
        "Record outcomes",
        list(STATUSES),
        default=list(STATUSES),
        key="bed_review_outcomes",
    )
    results = [r for r in report["results"] if r["status"] in statuses]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": r["record_id"],
                    "Tool": r["current_label"],
                    "Action": r["action"],
                    "Outcome": r["status"],
                    "Factual outcome": r["factual_status"],
                    "Error": r["error"],
                }
                for r in results
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if not results:
        return
    record_options = {f"{r['record_id']}: {r['proposed_label']}": r for r in results}
    selected = st.selectbox(
        "Inspect a record", list(record_options), key="bed_review_record"
    )
    if selected is None:
        st.error("Select a record to inspect.")
        return
    record = record_options[selected]
    check_statuses = st.multiselect(
        "Check outcomes",
        list(STATUSES),
        default=list(STATUSES),
        key="bed_review_check_outcomes",
    )
    visible_checks = {
        c["id"]: c for c in record["checks"] if c["status"] in check_statuses
    }
    if visible_checks:
        check_id = st.selectbox(
            "Inspect a check", list(visible_checks), key="bed_review_check"
        )
        if check_id is None:
            st.error("Select a check to inspect.")
        else:
            check = visible_checks[check_id]
            st.markdown(f"**{check['status']} ({check['kind']})**")
            st.write(check["claim"])
            st.write(check["reason"])
            for evidence in check["evidence"]:
                document = report["documents"][evidence["document_id"]]
                st.write(document["final_url"] or document["url"])
                st.caption(
                    f"Retrieved {document['retrieved_at']} | SHA-256 {document['content_sha256']}"
                )
                st.code(evidence["quote"], language=None)
    failures = [
        report["documents"][key]
        for key in record["source_ids"]
        if report["documents"][key]["status"] == "failed"
    ]
    if failures:
        st.warning(f"{len(failures)} source retrievals failed for this record.")
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    if record["omitted_source_urls"]:
        st.warning(
            "Source limit omitted some candidate URLs; this is not an exhaustive check."
        )
        st.json(record["omitted_source_urls"], expanded=False)


def render_review_verification(
    api: BuildEntityDataset, *, api_key: str, model: str
) -> None:
    with st.expander(
        "Verify review decisions (fresh evidence; no changes applied)", expanded=False
    ):
        st.caption(
            "Import proposed repairs, renames, splits, duplicate checks or scope reviews. "
            "New runs retrieve linked public HTML/text/PDF sources. This is not an exhaustive "
            "web search; browser-only or inaccessible pages may remain inconclusive. "
            "Loaded read-only datasets are supported. Nothing is merged, split or rewritten."
        )
        busy = api.is_running
        backend_options = {
            "OpenAI API": "openai",
            "Authenticated Copilot CLI (host account)": "copilot",
        }
        backend_choice = st.selectbox(
            "Verification backend",
            list(backend_options),
            disabled=busy,
            key="bed_review_backend",
        )
        if backend_choice is None:
            st.error("Select a verification backend.")
            return
        backend = backend_options[backend_choice]
        st.warning(
            "Verification sends proposal/record context and retrieved public text to the selected "
            "AI service. Copilot uses the account authenticated on the application host, not "
            "necessarily the browser user. Account usage/charges apply."
        )
        authorized = st.checkbox(
            "I authorize this verification and the selected account's usage",
            key="bed_review_authorized",
            disabled=busy,
        )
        concurrency = st.number_input(
            "Parallel record checks",
            min_value=1,
            max_value=12,
            value=4,
            key="bed_review_concurrency",
            disabled=busy,
        )
        max_sources = st.number_input(
            "Maximum linked sources per record",
            min_value=1,
            max_value=48,
            value=12,
            key="bed_review_max_sources",
            disabled=busy,
        )
        upload = st.file_uploader(
            "Review proposals (CSV or JSON)",
            type=["csv", "json"],
            key="bed_review_upload",
            disabled=busy,
        )
        selected_ids = st.text_input(
            "Record IDs to verify (optional)",
            placeholder="All proposal rows; e.g. 4,21,75",
            key="bed_review_ids",
            disabled=busy,
        )
        proposals = []
        if upload is not None:
            try:
                proposals = load_proposals(
                    upload.getvalue().decode("utf-8-sig"),
                    upload.name.rsplit(".", 1)[-1],
                )
                if selected_ids.strip():
                    ids = {int(s.strip()) for s in selected_ids.split(",")}
                    if ids - {p["record_id"] for p in proposals}:
                        msg = "Some selected IDs are absent from the proposal file"
                        raise ValueError(msg)  # noqa: TRY301
                    proposals = [p for p in proposals if p["record_id"] in ids]
                st.caption(
                    f"{len(proposals)} proposals selected; canonical IDs/labels and all aliases are validated at start."
                )
            except (ValueError, UnicodeError, KeyError) as exc:
                st.error(f"Invalid proposal file: {exc}")
                proposals = []
        if backend == "openai" and not api_key:
            st.info(
                "Configure an OpenAI key in Settings, or choose the authenticated host Copilot CLI."
            )
        if st.button(
            "Start fresh verification",
            key="bed_review_start",
            disabled=busy
            or not proposals
            or not authorized
            or (backend == "openai" and not api_key),
        ):
            try:
                api.start_review_verification(
                    proposals,
                    backend=backend,
                    api_key=api_key or None,
                    concurrency=int(concurrency),
                    max_sources=int(max_sources),
                )
                busy = api.is_running
            except (ValueError, OSError) as exc:
                st.error(f"Could not start verification: {exc}")

        saved = api.list_review_verifications()
        if saved and not busy:
            saved_options = {
                f"{i + 1}. {capture['started_at']} | {capture['backend']} | "
                f"{capture['summary']['completed_records']}/{capture['summary']['total_records']}": capture
                for i, capture in enumerate(saved)
            }
            choice = st.selectbox(
                "Saved verification captures",
                list(saved_options),
                key="bed_review_saved",
            )
            if choice is None:
                st.error("Select a saved verification capture.")
                return
            capture = saved_options[choice]
            retry_failed = st.checkbox(
                "Retry failed sources on resume (new timestamps; previous failures retained)",
                value=False,
                key="bed_review_retry_failed",
            )
            if st.button("Open saved verification", key="bed_review_open"):
                try:
                    api.open_review_verification(capture["directory"])
                except (OSError, ValueError) as exc:
                    st.error(f"Could not open verification: {exc}")
            if st.button(
                (
                    "Resume saved capture (retry failures; reassess affected records)"
                    if retry_failed
                    else "Resume saved capture (reuse original evidence timestamps)"
                ),
                key="bed_review_resume",
                disabled=not authorized
                or backend != capture["backend"]
                or (backend == "openai" and not api_key),
            ):
                try:
                    api.resume_review_verification(
                        capture["directory"],
                        api_key=api_key or None,
                        model=model if backend == "openai" else None,
                        concurrency=int(concurrency),
                        retry_failed_sources=retry_failed,
                    )
                    busy = api.is_running
                except (OSError, ValueError, KeyError) as exc:
                    st.error(f"Could not resume verification: {exc}")

        progress = api.review_verification_progress
        if api.is_verifying_review:
            st.info(f"{progress.stage}: {progress.current}/{progress.total}")
            if st.button(
                "Pause verification after in-flight checks", key="bed_review_stop"
            ):
                try:
                    api.stop_review_verification()
                    st.info(
                        "Pausing; in-flight requests will finish and checkpoints will be saved."
                    )
                except ValueError as exc:
                    st.info(str(exc))
        if progress.error:
            st.error(progress.error)
        report_upload = st.file_uploader(
            "Open an exported verification report (JSON)",
            type=["json"],
            key="bed_review_report_upload",
            disabled=busy,
        )
        report = api.review_verification_report
        if report_upload is not None:
            try:
                report = parse_report(report_upload.getvalue().decode("utf-8-sig"))
            except (ValueError, UnicodeError) as exc:
                st.error(f"Could not open exported report: {exc}")
                report = {}
        if report:
            if report["dataset_fingerprint"] != api.review_dataset_fingerprint:
                st.warning(
                    "STALE REPORT: this capture does not match the current dataset."
                )
            if not report["complete"]:
                st.warning("Partial capture: some records are still pending.")
            try:
                _render_report(report)
                st.download_button(
                    "Download verification JSON",
                    json.dumps(report, ensure_ascii=False, indent=2),
                    "verification.json",
                    "application/json",
                    key="bed_review_json",
                )
                st.download_button(
                    "Download record decisions CSV",
                    decisions_csv(report).encode("utf-8-sig"),
                    "verification_decisions.csv",
                    "text/csv",
                    key="bed_review_decisions_csv",
                )
                st.download_button(
                    "Download per-check CSV",
                    checks_csv(report).encode("utf-8-sig"),
                    "verification_checks.csv",
                    "text/csv",
                    key="bed_review_csv",
                )
                if (
                    not busy
                    and report_upload is None
                    and st.button(
                        "Prepare evidence ZIP",
                        key="bed_review_bundle_prepare",
                    )
                ):
                    st.session_state["bed_review_bundle"] = (
                        report["updated_at"],
                        api.review_verification_bundle(),
                    )
                bundle = st.session_state.get("bed_review_bundle")
                if (
                    bundle
                    and bundle[0] == report["updated_at"]
                    and report_upload is None
                ):
                    st.download_button(
                        "Download evidence ZIP",
                        bundle[1],
                        "verification_evidence.zip",
                        "application/zip",
                        key="bed_review_bundle_download",
                    )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                st.error(f"Report or evidence files are invalid/unavailable: {exc}")
        if api.is_verifying_review:
            time.sleep(2)
            st.rerun()
