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
                "Verify attribute values",
                value=sv.bed_verify.value,
                help="Run a verification pass to web-ground any unverified values.",
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
                # Live progress display
                st.markdown("#### Research in progress…")
                max_q = max(int(sv.bed_max_queries.value or 0), 1)
                frac = min(prog.query_count / max_q, 1.0)
                st.progress(
                    frac,
                    text=f"{prog.stage} — {prog.query_count}/{max_q} queries",
                )

                m1, m2 = st.columns(2)
                m1.metric("Entities found", prog.entity_count)
                m2.metric("Cost (USD)", f"${api.usage.total_cost_usd:.2f}")

                # Live dataset preview (built from the running record set).
                live_df = (
                    api.current_dataframe()
                    if hasattr(api, "current_dataframe")
                    else api.dataframe
                )
                if live_df is not None and not live_df.empty:
                    st.markdown(f"##### Dataset so far — {len(live_df)} entities")
                    st.dataframe(
                        live_df,
                        height=400,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("Dataset will appear here as entities are extracted…")

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
