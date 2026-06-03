# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

# ruff: noqa
import json
import os
import time

import pandas as pd
import streamlit as st

import app.util.example_outputs_ui as example_outputs_ui
import workflows.build_entity_dataset.variables as bed_variables
import workflows.build_entity_dataset.functions as functions
from util import ui_components
from util.session_variables import SessionVariables

from intelligence_toolkit.build_entity_dataset import config


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def _suggestion_key(s: dict) -> str:
    """Stable identity for an AI alias-merge suggestion so we can track
    dismissals across reruns."""
    primary = (s.get("primary") or "").strip().casefold()
    members = sorted(
        (m or "").strip().casefold() for m in (s.get("members") or [])
    )
    return primary + "::" + ",".join(members)


def _render_continue_research(api, sv) -> None:
    """Render a 'Continue research' control that extends the current run
    instead of discarding results. Used after both successful completion
    and user stop. No-op when the in-memory Schemify state can't be
    resumed (e.g. loaded purely from disk)."""
    if not api.can_continue_research():
        return

    api_key = functions.get_api_key()
    with st.expander(
        "Continue research (keep current results, run more queries)",
        expanded=False,
    ):
        st.caption(
            "Runs additional web search queries on top of the current "
            "dataset — useful after adding candidate entities, applying "
            "exclusions, or curating aliases. Existing records, attribute "
            "values and citations are preserved. The default plan biases "
            "toward filling in missing attributes for known entities."
        )
        c1, c2 = st.columns(2)
        more_q = c1.number_input(
            "Additional query budget",
            min_value=1,
            max_value=500,
            value=min(int(sv.bed_max_queries.value or 30), 30),
            key="bed_continue_max_q",
            help="Cap on the number of extra search queries to run.",
        )
        concur = c2.number_input(
            "Concurrency",
            min_value=1,
            max_value=20,
            value=int(sv.bed_concurrency.value or 5),
            key="bed_continue_concur",
        )
        verify = st.checkbox(
            "Also verify unverified attribute values",
            value=False,
            key="bed_continue_verify",
        )
        if st.button(
            "Continue research",
            type="primary",
            disabled=not api_key,
            key="bed_continue_btn",
        ):
            if not api_key:
                st.error("No API key found. Configure it in the Settings page.")
            else:
                ok = api.continue_research(
                    max_queries=int(more_q),
                    concurrency=int(concur),
                    verify=bool(verify),
                )
                if ok:
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(
                        "Can't continue this run — start a fresh research "
                        "task instead."
                    )


async def create(sv: bed_variables.SessionVariables, workflow=None):
    sv_home = SessionVariables("home")
    ui_components.check_ai_configuration()
    api = sv.workflow_object.value

    (
        intro_tab,
        define_tab,
        run_tab,
        review_tab,
        export_tab,
        examples_tab,
    ) = st.tabs(
        [
            "Build Entity Dataset workflow:",
            "Define task",
            "Run research",
            "Review dataset",
            "Export",
            "View example outputs",
        ]
    )

    # ── Intro ──────────────────────────────────────────────────
    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content)

    # ── Define task ────────────────────────────────────────────
    with define_tab:
        st.markdown("##### Task definition")
        st.markdown(
            "Describe the category of entities you want to discover. "
            "The workflow will search the web and extract structured records."
        )

        col_left, col_right = st.columns([3, 2])
        with col_left:
            sv.bed_category.value = st.text_input(
                "Entity category",
                value=sv.bed_category.value,
                placeholder="e.g. Open-source relational databases",
                help="What kind of entities should be discovered?",
            )
            sv.bed_guidance.value = st.text_area(
                "Guidance (optional)",
                value=sv.bed_guidance.value,
                placeholder="e.g. Focus on systems with active communities and >1000 GitHub stars.",
                height=100,
                help="Additional instructions to steer extraction quality and scope.",
            )
            st.markdown("##### Schema attributes (optional)")
            st.markdown(
                "Provide a JSON list of attribute definitions to use instead of the "
                "auto-generated schema. Leave blank to let the model propose a schema."
            )
            schema_placeholder = json.dumps(
                [
                    {"name": "founded_year", "description": "Year the project was founded"},
                    {"name": "primary_language", "description": "Main programming language", "is_closed_set": True},
                ],
                indent=2,
            )
            sv.bed_schema_json.value = st.text_area(
                "Schema JSON",
                value=sv.bed_schema_json.value,
                placeholder=schema_placeholder,
                height=200,
                help="Optional list of {name, description, is_closed_set, is_multi_valued} objects.",
            )

        with col_right:
            st.markdown("##### Research configuration")
            sv.bed_model.value = st.selectbox(
                "Model",
                ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"],
                index=["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"].index(sv.bed_model.value)
                if sv.bed_model.value in ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]
                else 1,
                help="Model used for both search and extraction.",
            )
            sv.bed_max_queries.value = st.number_input(
                "Max queries",
                min_value=5,
                max_value=500,
                value=sv.bed_max_queries.value,
                step=5,
                help="Total web-search queries. More queries = more entities, higher cost.",
            )
            sv.bed_concurrency.value = st.number_input(
                "Concurrency",
                min_value=1,
                max_value=20,
                value=sv.bed_concurrency.value,
                help="Parallel web searches. Higher values are faster but may hit rate limits.",
            )
            sv.bed_budget.value = st.number_input(
                "Budget (USD)",
                min_value=0.5,
                max_value=500.0,
                value=sv.bed_budget.value,
                step=0.5,
                help="Maximum spend on LLM/search calls.",
            )
            sv.bed_verify.value = st.checkbox(
                "Auto-verify attribute values after research",
                value=sv.bed_verify.value,
                help=(
                    "If on, runs a web-grounded verification pass at the end of "
                    "research. Verification can also be triggered manually from "
                    "the Review dataset tab."
                ),
            )

        if sv.bed_category.value:
            if st.button(
                "Save task definition",
                type="primary",
                help="Save settings and continue to the Run research tab.",
            ):
                st.success("Task saved. Switch to the **Run research** tab to start.")

    # ── Run research ───────────────────────────────────────────
    with run_tab:
        if not sv.bed_category.value:
            st.info("Define a task in the **Define task** tab first.")
        else:
            st.markdown(f"**Category:** {sv.bed_category.value}")
            if sv.bed_guidance.value:
                st.markdown(f"**Guidance:** {sv.bed_guidance.value}")

            st.divider()
            prog = api.progress

            if api.is_running:
                # Pull latest live counters from schemify before rendering.
                refresh = getattr(api, "refresh_progress", None)
                if callable(refresh):
                    refresh()

                # Build a stable layout once per rerun: header → progress
                # bar → metrics → live dataframe → stop button. Keeping
                # the structure identical between reruns prevents the
                # Streamlit reconciler from tearing down and re-creating
                # blocks, which is what causes the visible flicker.
                st.markdown("#### Research in progress…")
                progress_slot = st.empty()
                metrics_slot = st.container()
                table_slot = st.container()
                button_slot = st.container()

                if prog.stage.startswith("Verifying") and prog.total:
                    frac = min(prog.current / max(prog.total, 1), 1.0)
                    progress_slot.progress(
                        frac,
                        text=f"{prog.stage} ({prog.current}/{prog.total})",
                    )
                else:
                    max_q = max(int(sv.bed_max_queries.value or 0), 1)
                    frac = min(prog.query_count / max_q, 1.0)
                    progress_slot.progress(
                        frac,
                        text=f"{prog.stage} — {prog.query_count}/{max_q} queries",
                    )

                with metrics_slot:
                    m1, m2 = st.columns(2)
                    m1.metric("Entities found", prog.entity_count)
                    m2.metric("Cost (USD)", f"${api.usage.total_cost_usd:.2f}")

                # Live dataset preview (built from the running record set).
                live_df = (
                    api.current_dataframe()
                    if hasattr(api, "current_dataframe")
                    else api.dataframe
                )
                with table_slot:
                    # Always render the header + table block (even when
                    # empty) so the layout doesn't reshuffle once the
                    # first entities arrive.
                    count = 0 if live_df is None else len(live_df)
                    st.markdown(f"##### Dataset so far — {count} entities")
                    if live_df is not None and not live_df.empty:
                        st.dataframe(
                            live_df,
                            height=400,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption(
                            "Entities will appear here as they are extracted…"
                        )

                with button_slot:
                    if st.button("Stop and save current results"):
                        api.stop_research()
                        st.rerun()

                # Auto-refresh every 2 s while running
                time.sleep(2.0)
                st.rerun()

            elif prog.is_complete:
                st.success(
                    f"Research complete — **{prog.entity_count} entities** discovered."
                )
                m1, m2 = st.columns(2)
                m1.metric("Total queries", api.usage.queries_run)
                m2.metric("Estimated cost (USD)", f"${api.usage.total_cost_usd:.2f}")
                st.info("Switch to the **Review dataset** or **Export** tab.")

                _render_continue_research(api, sv)

                if st.button("Re-run research (discard current results)"):
                    api.reset()
                    st.rerun()

            elif prog.error:
                st.error(f"Research failed: {prog.error}")
                if st.button("Reset and retry"):
                    api.reset()
                    st.rerun()

            elif prog.stage == "Stopped by user":
                st.warning("Research stopped. Partial results are available.")
                st.info("Switch to the **Review dataset** or **Export** tab.")

                _render_continue_research(api, sv)

            else:
                # Not started yet
                st.markdown(
                    "Click **Start research** to begin web search and entity extraction. "
                    "This may take several minutes depending on the number of queries."
                )

                api_key = functions.get_api_key()

                # Parse schema if provided
                schema_attrs = None
                if sv.bed_schema_json.value.strip():
                    try:
                        schema_attrs = json.loads(sv.bed_schema_json.value)
                        st.caption(f"Using {len(schema_attrs)} predefined schema attributes.")
                    except json.JSONDecodeError as e:
                        st.error(f"Schema JSON is invalid: {e}")
                        schema_attrs = None

                if st.button("Start research", type="primary", disabled=not api_key):
                    if not api_key:
                        st.error("No API key found. Configure it in the Settings page.")
                    else:
                        api.start_research(
                            api_key=api_key,
                            category=sv.bed_category.value,
                            guidance=sv.bed_guidance.value,
                            schema_attributes=schema_attrs,
                            max_queries=sv.bed_max_queries.value,
                            concurrency=sv.bed_concurrency.value,
                            model=sv.bed_model.value,
                            budget=sv.bed_budget.value,
                            verify=sv.bed_verify.value,
                        )
                        time.sleep(0.3)
                        st.rerun()

                # ── Resume from a previously completed run ────────
                saved_runs = api.list_saved_runs()
                if saved_runs:
                    st.divider()
                    st.markdown("##### Resume a previous run")
                    labels = [
                        f"{r['timestamp']} — {r['category'] or '(unknown)'} "
                        f"· {r['entity_count']} entities · ${r['total_cost_usd']:.2f}"
                        for r in saved_runs
                    ]
                    idx = st.selectbox(
                        "Saved runs",
                        options=list(range(len(saved_runs))),
                        format_func=lambda i: labels[i],
                        key="bed_resume_idx",
                    )
                    if st.button("Resume selected run"):
                        try:
                            api.load_saved_run(saved_runs[idx]["path"])
                            st.success(
                                f"Loaded {saved_runs[idx]['entity_count']} entities."
                            )
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed to load run: {e}")

    # ── Review dataset ─────────────────────────────────────────
    with review_tab:
        df = api.dataframe
        if df is None or df.empty:
            st.info("No dataset yet. Run research first.")
        else:
            st.markdown(f"##### Dataset — {len(df)} entities")
            st.dataframe(df, height=500, use_container_width=True, hide_index=True)

            schema = api.schema_attributes
            if schema:
                with st.expander("Schema attributes", expanded=False):
                    schema_df = pd.DataFrame(schema)
                    st.dataframe(schema_df, use_container_width=True, hide_index=True)

            # ── Verification (on demand) ──────────────────────
            count_fn = getattr(api, "count_unverified", None)
            unverified_entities, unverified_values = (
                count_fn() if callable(count_fn) else (0, 0)
            )
            with st.expander(
                f"Verify attribute values ({unverified_values} unverified across "
                f"{unverified_entities} entities)",
                expanded=False,
            ):
                if unverified_values == 0:
                    st.success("All attribute values are already web-sourced.")
                else:
                    st.markdown(
                        "Verification runs one targeted web search per entity "
                        "with unsourced values. This may take a few minutes and "
                        "incur extra LLM/search cost."
                    )
                    if api.is_running:
                        st.info(f"Busy: {api.progress.stage}")
                    else:
                        if st.button(
                            f"Verify {unverified_values} unverified values",
                            type="primary",
                            key="bed_verify_btn",
                        ):
                            api.start_verification(
                                concurrency=max(
                                    12, int(sv.bed_concurrency.value or 12)
                                )
                            )
                            st.rerun()

            # ── Normalize attribute values (J) ────────────────
            schema_names = [sa.get("name") for sa in schema] if schema else []
            with st.expander("Normalize attribute values", expanded=False):
                st.caption(
                    "Cluster near-duplicate values and map them to canonical "
                    "forms. Choose specific attributes or normalize all."
                )
                picks = st.multiselect(
                    "Attributes to normalize (empty = all)",
                    options=schema_names,
                    key="bed_norm_attrs",
                )
                if api.is_running:
                    st.info(f"Busy: {api.progress.stage}")
                else:
                    if st.button("Run normalization", key="bed_norm_btn"):
                        api.start_normalize(attributes=picks or None)
                        st.rerun()

            # ── Schema editor (K) ─────────────────────────────
            with st.expander("Edit schema attributes", expanded=False):
                if not schema_names:
                    st.caption("No schema attributes yet.")
                else:
                    target_attr = st.selectbox(
                        "Attribute",
                        options=schema_names,
                        key="bed_edit_attr",
                    )
                    new_name = st.text_input(
                        "Rename to",
                        value=target_attr,
                        key="bed_edit_newname",
                    )
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        if st.button("Apply rename", key="bed_rename_btn"):
                            n = api.rename_attribute(target_attr, new_name)
                            st.success(f"Renamed in {n} record fields.")
                            st.rerun()
                    with col_e2:
                        if st.button(
                            "Remove attribute",
                            key="bed_remove_btn",
                            type="secondary",
                        ):
                            n = api.remove_attribute(target_attr)
                            st.success(f"Removed from {n} record fields.")
                            st.rerun()

            # ── Add candidate entities (L) ────────────────────
            with st.expander("Add candidate entities & guidance", expanded=False):
                st.caption(
                    "Seed labels you want included in the dataset (they will "
                    "be added as blank records — run research again to expand "
                    "their attributes) and/or leave a free-form note that "
                    "future research and verification prompts will see."
                )
                cand_text = st.text_area(
                    "Candidate names (one per line or comma-separated)",
                    key="bed_candidates_text",
                    height=120,
                )
                cand_file = st.file_uploader(
                    "…or upload a file (.txt, .csv, .tsv, .json)",
                    type=["txt", "csv", "tsv", "json"],
                    key="bed_candidates_file",
                )
                note_text = st.text_area(
                    "Comment / additional guidance for the research agent",
                    key="bed_candidates_note",
                    height=100,
                    placeholder=(
                        "e.g. Focus on Southeast Asian member states only, "
                        "or: Exclude organisations dissolved before 2010."
                    ),
                )
                if st.button("Apply", key="bed_candidates_btn"):
                    names: list[str] = [
                        tok.strip()
                        for tok in cand_text.replace(",", "\n").splitlines()
                        if tok.strip()
                    ]
                    if cand_file is not None:
                        names.extend(
                            api.parse_candidate_file(
                                cand_file.name, cand_file.getvalue()
                            )
                        )
                    added = api.add_candidate_entities(names) if names else 0
                    noted = api.append_guidance(note_text)
                    msgs = []
                    if added:
                        msgs.append(f"Added {added} new entities.")
                    elif names:
                        msgs.append("No new entities added (duplicates).")
                    if noted:
                        msgs.append("Guidance updated.")
                    if msgs:
                        st.success(" ".join(msgs))
                        st.rerun()
                    else:
                        st.info("Nothing to apply.")

            # ── Curate entity groups (M) ──────────────────────
            alias_groups_all = (
                api.list_alias_groups() if hasattr(api, "list_alias_groups") else []
            )
            grouped = [g for g in alias_groups_all if g.get("aliases")]
            curate_title = (
                f"Curate entity groups ({len(grouped)} of {len(alias_groups_all)} "
                "entities have aliases)"
            )
            with st.expander(curate_title, expanded=False):
                st.caption(
                    "Fuzzy deduplication may have over-grouped entities — "
                    "e.g. a category-like phrase becoming the canonical "
                    "name for specific tools. Review groups below, set a "
                    "better canonical, remove or split out spurious "
                    "aliases, or manually merge two entries. Splits are "
                    "remembered so later passes won't re-merge them."
                )

                if not alias_groups_all:
                    st.info("No entities yet.")
                else:
                    # Quick filter so big datasets stay navigable.
                    show_filter = st.text_input(
                        "Filter by name or alias (case-insensitive)",
                        key="bed_curate_filter",
                        placeholder="e.g. memex",
                    )
                    only_groups = st.checkbox(
                        "Only show entities with aliases",
                        value=True,
                        key="bed_curate_only_grouped",
                    )
                    needle = (show_filter or "").strip().lower()
                    candidates = grouped if only_groups else alias_groups_all
                    if needle:
                        def _matches(g: dict) -> bool:
                            hay = " ".join(
                                [g.get("label") or ""] + list(g.get("aliases") or [])
                            ).lower()
                            return needle in hay
                        candidates = [g for g in candidates if _matches(g)]

                    if not candidates:
                        st.info("No entities match the current filter.")
                    else:
                        st.caption(f"Showing {min(len(candidates), 50)} of {len(candidates)} entities.")
                        for ci, g in enumerate(candidates[:50]):
                            label = g["label"]
                            aliases = g.get("aliases") or []
                            counts = g.get("alias_counts") or {}
                            with st.container(border=True):
                                hdr_cols = st.columns([6, 2])
                                hdr_cols[0].markdown(
                                    f"**{label}**" + (
                                        f"  &nbsp;·&nbsp; {len(aliases)} alias"
                                        f"{'es' if len(aliases) != 1 else ''}"
                                        if aliases else ""
                                    )
                                )
                                if counts:
                                    summary = ", ".join(
                                        f"{k} ({v})"
                                        for k, v in sorted(
                                            counts.items(), key=lambda kv: -kv[1]
                                        )[:6]
                                    )
                                    hdr_cols[0].caption(f"variants seen: {summary}")
                                hdr_cols[1].caption(
                                    f"total mentions: {g.get('total_count', 1)}"
                                )

                                if aliases:
                                    # Canonical reassignment
                                    canon_options = [label] + aliases
                                    new_canon = st.selectbox(
                                        "Canonical label",
                                        options=canon_options,
                                        index=0,
                                        key=f"bed_curate_canon_{ci}",
                                    )
                                    custom_canon = st.text_input(
                                        "…or type a new canonical name",
                                        key=f"bed_curate_canon_custom_{ci}",
                                        placeholder="(leave empty to use selection above)",
                                    )
                                    set_btn, _spacer = st.columns([2, 6])
                                    if set_btn.button(
                                        "Set canonical",
                                        key=f"bed_curate_set_canon_{ci}",
                                    ):
                                        target = (custom_canon or new_canon or "").strip()
                                        if target and target != label:
                                            if api.rename_record_canonical(label, target):
                                                st.success(
                                                    f"Canonical set to **{target}**."
                                                )
                                                st.rerun()
                                            else:
                                                st.error("Could not update canonical.")
                                        else:
                                            st.info("No change.")

                                    # Per-alias controls
                                    st.caption("Aliases — split out, remove, or keep:")
                                    for ai, alias in enumerate(aliases):
                                        ac1, ac2, ac3 = st.columns([6, 2, 2])
                                        ac1.markdown(
                                            f"&nbsp;&nbsp;·&nbsp;`{alias}`"
                                            + (
                                                f" — {counts.get(alias, 1)} mention"
                                                f"{'s' if counts.get(alias, 1) != 1 else ''}"
                                                if alias in counts
                                                else ""
                                            ),
                                            unsafe_allow_html=True,
                                        )
                                        if ac2.button(
                                            "Split out",
                                            key=f"bed_curate_split_{ci}_{ai}",
                                            help=(
                                                "Promote this alias into its own "
                                                "record and remember not to "
                                                "re-merge them."
                                            ),
                                        ):
                                            if api.split_alias(label, alias):
                                                st.success(
                                                    f"Split **{alias}** into a new record."
                                                )
                                                st.rerun()
                                            else:
                                                st.error("Split failed.")
                                        if ac3.button(
                                            "Remove",
                                            key=f"bed_curate_rmalias_{ci}_{ai}",
                                            help=(
                                                "Drop this alias entirely — does "
                                                "not create a new record."
                                            ),
                                        ):
                                            if api.remove_alias(label, alias):
                                                st.success(f"Removed alias **{alias}**.")
                                                st.rerun()
                                            else:
                                                st.error("Remove failed.")
                                else:
                                    st.caption("No aliases — no curation needed.")

                    st.markdown("---")
                    st.markdown("**Merge two entities manually**")
                    all_labels = [g["label"] for g in alias_groups_all]
                    if len(all_labels) >= 2:
                        mc1, mc2, mc3 = st.columns([4, 4, 2])
                        primary_pick = mc1.selectbox(
                            "Keep this as canonical",
                            options=all_labels,
                            key="bed_curate_merge_primary",
                        )
                        other_options = [l for l in all_labels if l != primary_pick]
                        other_pick = mc2.selectbox(
                            "Merge this one in",
                            options=other_options,
                            key="bed_curate_merge_other",
                        )
                        if mc3.button("Merge", key="bed_curate_merge_btn"):
                            if api.merge_records(primary_pick, other_pick):
                                st.success(
                                    f"Merged **{other_pick}** → **{primary_pick}**."
                                )
                                st.rerun()
                            else:
                                st.error("Merge failed.")

                    # Show any do-not-merge constraints already in force.
                    dnm = (
                        api.do_not_merge_pairs
                        if hasattr(api, "do_not_merge_pairs")
                        else []
                    )
                    if dnm:
                        st.markdown("---")
                        st.caption(
                            f"{len(dnm)} pair(s) marked as 'do not merge' "
                            "(prevents re-merging by fuzzy passes):"
                        )
                        for pair in dnm[:25]:
                            st.markdown(
                                f"&nbsp;&nbsp;·&nbsp;`{pair[0]}` &nbsp;⇎&nbsp; "
                                f"`{pair[1]}`",
                                unsafe_allow_html=True,
                            )

                    st.markdown("---")
                    st.markdown("**Suggest groupings with AI**")
                    st.caption(
                        "Ask the LLM to review the current canonical labels "
                        "and propose any that should be merged into a "
                        "single record. Each proposal is shown for you to "
                        "accept or dismiss — nothing is applied "
                        "automatically."
                    )

                    suggestions = list(sv.bed_alias_suggestions.value or [])
                    dismissed = set(sv.bed_alias_dismissed.value or [])
                    visible_sugs = [
                        s for s in suggestions
                        if _suggestion_key(s) not in dismissed
                    ]
                    sc_run, sc_clear = st.columns([3, 1])
                    if sc_run.button(
                        "Suggest groupings (AI)",
                        key="bed_curate_ai_suggest_btn",
                        type="primary",
                    ):
                        with st.spinner("Asking the model…"):
                            try:
                                new_sugs = api.suggest_alias_groups()
                            except Exception as e:  # noqa: BLE001
                                new_sugs = None
                                st.error(f"Suggest failed: {e}")
                        if new_sugs is not None:
                            sv.bed_alias_suggestions.value = new_sugs
                            sv.bed_alias_dismissed.value = []
                            st.rerun()
                    if sc_clear.button(
                        "Clear suggestions", key="bed_curate_ai_clear"
                    ):
                        sv.bed_alias_suggestions.value = []
                        sv.bed_alias_dismissed.value = []
                        st.rerun()

                    if not suggestions:
                        st.caption("No AI suggestions yet.")
                    elif not visible_sugs:
                        st.success(
                            "All suggestions have been actioned or dismissed."
                        )
                    else:
                        if len(visible_sugs) > 1:
                            apply_all_col, _ = st.columns([3, 5])
                            if apply_all_col.button(
                                f"Accept all {len(visible_sugs)} suggestions",
                                key="bed_curate_ai_accept_all",
                            ):
                                applied_total = 0
                                for s in visible_sugs:
                                    applied_total += api.apply_alias_suggestion(
                                        s.get("primary", ""),
                                        s.get("members", []),
                                    )
                                sv.bed_alias_suggestions.value = []
                                sv.bed_alias_dismissed.value = []
                                st.success(
                                    f"Applied {applied_total} merge(s)."
                                )
                                st.rerun()

                        for si, s in enumerate(visible_sugs):
                            primary = s.get("primary", "")
                            members = list(s.get("members", []) or [])
                            reason = s.get("reason", "")
                            with st.container(border=True):
                                st.markdown(
                                    f"Merge into **{primary}** ← " +
                                    ", ".join(f"`{m}`" for m in members)
                                )
                                if reason:
                                    st.caption(reason)
                                ab1, ab2 = st.columns(2)
                                if ab1.button(
                                    "Accept",
                                    key=f"bed_curate_ai_accept_{si}",
                                    type="primary",
                                ):
                                    n = api.apply_alias_suggestion(primary, members)
                                    sv.bed_alias_suggestions.value = [
                                        x for x in suggestions
                                        if _suggestion_key(x) != _suggestion_key(s)
                                    ]
                                    st.success(f"Merged {n} record(s) into {primary}.")
                                    st.rerun()
                                if ab2.button(
                                    "Dismiss",
                                    key=f"bed_curate_ai_dismiss_{si}",
                                ):
                                    dismissed.add(_suggestion_key(s))
                                    sv.bed_alias_dismissed.value = list(dismissed)
                                    st.rerun()

            # ── Exclusions ────────────────────────────────────
            current_exclusions = (
                api.exclusions if hasattr(api, "exclusions") else []
            )
            with st.expander(
                f"Exclude entities ({len(current_exclusions)} rule"
                f"{'' if len(current_exclusions) == 1 else 's'})",
                expanded=False,
            ):
                st.caption(
                    "Define rules for entities to leave out of this dataset. "
                    "Rules can target a single named entity or any entity "
                    "matching an attribute condition (e.g. *Currency is "
                    "missing*, *Continent equals Africa*). Each rule's reason "
                    "is injected into future discovery and verification "
                    "prompts so the agent learns to skip similar cases."
                )

                if current_exclusions:
                    for i, rule in enumerate(current_exclusions):
                        c1, c2 = st.columns([10, 1])
                        if rule.get("attribute"):
                            op = (rule.get("operator") or "equals").lower()
                            vals = rule.get("values") or []
                            if op == "missing":
                                summary = f"`{rule['attribute']}` is missing"
                            elif op in ("equals", "contains", "regex") and vals:
                                summary = (
                                    f"`{rule['attribute']}` {op} \"{vals[0]}\""
                                )
                            elif op == "in" and vals:
                                summary = (
                                    f"`{rule['attribute']}` in [{', '.join(vals)}]"
                                )
                            else:
                                summary = f"`{rule['attribute']}` {op}"
                        else:
                            summary = f"**{rule.get('label','')}**"
                        c1.markdown(summary)
                        reason = rule.get("reason", "")
                        if reason:
                            c1.caption(reason)
                        if c2.button("✕", key=f"bed_excl_rm_{i}"):
                            api.remove_exclusion(rule)
                            st.rerun()
                    st.markdown("---")

                schema_names = [sa.get("name") for sa in schema] if schema else []
                rule_kind = st.radio(
                    "Rule type",
                    options=["Named entity", "Attribute condition"],
                    horizontal=True,
                    key="bed_excl_kind",
                )

                if rule_kind == "Named entity":
                    excl_label = st.text_input(
                        "Entity to exclude", key="bed_excl_label"
                    )
                    excl_reason = st.text_input(
                        "Reason (optional)",
                        key="bed_excl_reason_lbl",
                        placeholder="e.g. not a sovereign state",
                    )
                    excl_drop = st.checkbox(
                        "Also remove matching record from dataset if present",
                        value=True,
                        key="bed_excl_drop_lbl",
                    )
                    if st.button("Add exclusion", key="bed_excl_add_lbl_btn"):
                        added, removed = api.add_label_exclusion(
                            excl_label, excl_reason, remove_existing=excl_drop
                        )
                        if added:
                            msg = f"Exclusion added for **{excl_label}**."
                            if removed:
                                msg += f" Removed {removed} matching record(s)."
                            st.success(msg)
                            st.rerun()
                        else:
                            st.info("Provide an entity label first.")
                else:
                    if not schema_names:
                        st.caption("No schema attributes available yet.")
                    else:
                        attr_pick = st.selectbox(
                            "Attribute", options=schema_names, key="bed_excl_attr"
                        )
                        op_pick = st.selectbox(
                            "Operator",
                            options=[
                                "missing",
                                "equals",
                                "in",
                                "contains",
                                "regex",
                            ],
                            key="bed_excl_op",
                            help=(
                                "missing: value is empty or 'N/A'. "
                                "equals: exact match. in: any of a list "
                                "(comma-separated). contains: substring. "
                                "regex: Python regex (case-insensitive)."
                            ),
                        )
                        excl_vals: list[str] = []
                        if op_pick != "missing":
                            raw_vals = st.text_input(
                                "Value(s) — comma-separated for 'in'",
                                key="bed_excl_vals",
                            )
                            excl_vals = [
                                v.strip()
                                for v in (raw_vals or "").split(",")
                                if v.strip()
                            ]
                        excl_reason_a = st.text_input(
                            "Reason (optional)",
                            key="bed_excl_reason_attr",
                            placeholder="e.g. out of scope for this study",
                        )
                        excl_drop_a = st.checkbox(
                            "Also remove matching records from dataset",
                            value=True,
                            key="bed_excl_drop_attr",
                        )
                        if st.button("Add exclusion", key="bed_excl_add_attr_btn"):
                            added, removed = api.add_attribute_exclusion(
                                attr_pick,
                                op_pick,
                                excl_vals,
                                excl_reason_a,
                                remove_existing=excl_drop_a,
                            )
                            if added:
                                summary = (
                                    f"`{attr_pick}` is missing"
                                    if op_pick == "missing"
                                    else f"`{attr_pick}` {op_pick} {excl_vals}"
                                )
                                msg = f"Exclusion added: {summary}."
                                if removed:
                                    msg += f" Removed {removed} matching record(s)."
                                st.success(msg)
                                st.rerun()
                            else:
                                st.info(
                                    "Provide values for this operator first."
                                )

            # ── Harmful content scan (I) ──────────────────────
            findings_state = list(sv.bed_safety_findings.value or [])
            dismissed_state = set(sv.bed_safety_dismissed.value or [])
            visible_findings = [
                f for f in findings_state
                if (f.get("label") or "") not in dismissed_state
            ]
            scan_title = "Scan for harmful content"
            if visible_findings:
                scan_title += f" ({len(visible_findings)} flagged)"
            with st.expander(scan_title, expanded=False):
                st.caption(
                    "Use the LLM to flag records that contain potentially "
                    "harmful, sensitive, or unsafe content. The prompt below "
                    "is editable — adjust the categories or instructions to "
                    "match your safety policy. One request is sent per entity."
                )

                default_prompt = api.DEFAULT_SAFETY_PROMPT
                current_prompt = sv.bed_safety_prompt.value or default_prompt
                edited = st.text_area(
                    "Safety classifier prompt",
                    value=current_prompt,
                    height=220,
                    key="bed_safety_prompt_ta",
                    help=(
                        "Must contain the placeholder `{record}`, which is "
                        "replaced with the entity's fields. The classifier "
                        "should reply `SAFE` for benign records or a "
                        "comma-separated category list followed by a "
                        "one-sentence reason."
                    ),
                )
                sv.bed_safety_prompt.value = edited

                c_run, c_reset, c_clear = st.columns([2, 1, 1])
                if c_run.button("Run safety scan", key="bed_scan_btn", type="primary"):
                    with st.spinner("Scanning…"):
                        try:
                            new_findings = api.scan_harmful_content(
                                prompt_template=edited
                            )
                        except Exception as e:  # noqa: BLE001
                            new_findings = None
                            st.error(f"Scan failed: {e}")
                    if new_findings is not None:
                        sv.bed_safety_findings.value = new_findings
                        sv.bed_safety_dismissed.value = []
                        st.rerun()
                if c_reset.button("Reset prompt", key="bed_scan_reset"):
                    sv.bed_safety_prompt.value = default_prompt
                    st.rerun()
                if c_clear.button("Clear results", key="bed_scan_clear"):
                    sv.bed_safety_findings.value = []
                    sv.bed_safety_dismissed.value = []
                    st.rerun()

                if findings_state and not visible_findings:
                    st.success(
                        "All flagged records have been resolved or dismissed."
                    )
                elif visible_findings:
                    st.warning(
                        f"{len(visible_findings)} record(s) flagged. "
                        "Choose an action for each."
                    )
                    for fi, finding in enumerate(visible_findings):
                        label = finding.get("label") or "(unlabeled)"
                        cats = ", ".join(finding.get("categories") or []) or "?"
                        reason = finding.get("reason") or ""
                        with st.container(border=True):
                            st.markdown(f"**{label}** — *{cats}*")
                            if reason:
                                st.caption(reason)
                            with st.expander("Show record fields", expanded=False):
                                fields = finding.get("fields") or {}
                                for k, v in fields.items():
                                    st.markdown(f"- **{k}**: {v}")

                            a1, a2, a3 = st.columns(3)
                            if a1.button(
                                "Remove record",
                                key=f"bed_scan_drop_{fi}",
                                help="Delete this entity from the dataset.",
                            ):
                                api.remove_record_by_label(label)
                                sv.bed_safety_findings.value = [
                                    f for f in findings_state
                                    if f.get("label") != label
                                ]
                                st.rerun()
                            if a2.button(
                                "Add as exclusion",
                                key=f"bed_scan_excl_{fi}",
                                help=(
                                    "Remove from dataset AND register a "
                                    "rule so future discovery skips it."
                                ),
                            ):
                                excl_reason = (
                                    f"Safety scan: {cats}"
                                    + (f" — {reason}" if reason else "")
                                )
                                api.add_label_exclusion(
                                    label, excl_reason, remove_existing=True
                                )
                                sv.bed_safety_findings.value = [
                                    f for f in findings_state
                                    if f.get("label") != label
                                ]
                                st.rerun()
                            if a3.button(
                                "Dismiss",
                                key=f"bed_scan_dismiss_{fi}",
                                help="Ignore this finding (keep the record).",
                            ):
                                dismissed_state.add(label)
                                sv.bed_safety_dismissed.value = list(dismissed_state)
                                st.rerun()

            if api.dataset_json:
                with st.expander("Load previously saved dataset JSON", expanded=False):
                    uploaded = st.file_uploader(
                        "Upload data.json", type=["json"], key="bed_load_json"
                    )
                    if uploaded and st.button("Load dataset"):
                        try:
                            data = json.load(uploaded)
                            api.load_dataset(data)
                            st.success("Dataset loaded.")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed to load: {e}")
            else:
                with st.expander("Load a previously saved dataset JSON", expanded=False):
                    uploaded = st.file_uploader(
                        "Upload data.json", type=["json"], key="bed_load_json2"
                    )
                    if uploaded and st.button("Load dataset"):
                        try:
                            data = json.load(uploaded)
                            api.load_dataset(data)
                            st.success("Dataset loaded.")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed to load: {e}")

    # ── Export ─────────────────────────────────────────────────
    with export_tab:
        df = api.dataframe
        if df is None or df.empty:
            st.info("No dataset yet. Run research first.")
        else:
            st.markdown("#### Download dataset")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download data.json",
                    data=api.get_dataset_bytes_json(),
                    file_name="data.json",
                    mime="application/json",
                )
            with col_dl2:
                st.download_button(
                    "Download data.csv",
                    data=api.get_dataset_bytes_csv(),
                    file_name="data.csv",
                    mime="text/csv",
                )

            st.divider()
            st.markdown("#### Download themed web interface")
            st.markdown(
                "Configure branding below, then download a self-contained HTML dashboard "
                "that lets anyone explore the dataset in a browser (no server needed)."
            )

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                sv.bed_title.value = st.text_input(
                    "Dashboard title",
                    value=sv.bed_title.value or sv.bed_category.value,
                    placeholder="My Entity Dataset",
                )
                sv.bed_subtitle.value = st.text_input(
                    "Subtitle / organisation",
                    value=sv.bed_subtitle.value,
                    placeholder="Produced by Contoso Research",
                )
                sv.bed_dataset_label.value = st.text_input(
                    "Dataset label (plural noun)",
                    value=sv.bed_dataset_label.value,
                    placeholder="entities",
                )
            with tcol2:
                sv.bed_primary_color.value = st.color_picker(
                    "Primary color",
                    value=sv.bed_primary_color.value,
                )
                sv.bed_accent_color.value = st.color_picker(
                    "Accent color",
                    value=sv.bed_accent_color.value,
                )
                logo_file = st.file_uploader(
                    "Logo image (optional)",
                    type=["png", "jpg", "jpeg", "svg"],
                    key="bed_logo_upload",
                )

            st.markdown("**Views to include**")
            v1, v2, v3 = st.columns(3)
            show_table = v1.checkbox(
                "Table", value=sv.bed_view_table.value, key="bed_view_table_cb"
            )
            show_cards = v2.checkbox(
                "Cards", value=sv.bed_view_cards.value, key="bed_view_cards_cb"
            )
            show_network = v3.checkbox(
                "Network", value=sv.bed_view_network.value, key="bed_view_network_cb"
            )
            sv.bed_view_table.value = show_table
            sv.bed_view_cards.value = show_cards
            sv.bed_view_network.value = show_network
            selected_views = [
                name for name, on in (
                    ("table", show_table),
                    ("cards", show_cards),
                    ("network", show_network),
                ) if on
            ]
            if not selected_views:
                st.caption(
                    "At least one view is required — defaulting to Table."
                )
                selected_views = ["table"]

            logo_bytes = logo_file.read() if logo_file else None
            logo_filename = logo_file.name if logo_file else None

            if st.button("Build & download web interface", type="primary"):
                if not sv.bed_title.value:
                    st.error("Please enter a dashboard title.")
                else:
                    try:
                        zip_bytes = api.build_dashboard_zip(
                            title=sv.bed_title.value,
                            subtitle=sv.bed_subtitle.value,
                            dataset_label=sv.bed_dataset_label.value or "entities",
                            primary_color=sv.bed_primary_color.value,
                            accent_color=sv.bed_accent_color.value,
                            logo_bytes=logo_bytes,
                            logo_filename=logo_filename,
                            views=selected_views,
                        )
                        st.download_button(
                            "Save dashboard.zip",
                            data=zip_bytes,
                            file_name="dashboard.zip",
                            mime="application/zip",
                        )
                        st.caption(
                            "Extract the ZIP and open `dashboard/dashboard.html` in a browser."
                        )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Failed to build dashboard: {e}")

    # ── Example outputs ────────────────────────────────────────
    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)
