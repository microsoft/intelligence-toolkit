# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import io
import os

import altair as alt
import numpy as np
import pandas as pd
import PIL
import streamlit as st
from st_aggrid import (
    AgGrid,
    ColumnsAutoSizeMode,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)

import app.util.example_outputs_ui as example_outputs_ui
import app.workflows.detect_case_patterns.variables as ap_variables
import toolkit.detect_case_patterns.config as config
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from toolkit.AI.classes import LLMCallback
from toolkit.detect_case_patterns import prompts


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: ap_variables.SessionVariables, workflow):
    ui_components.check_ai_configuration()

    dcp = sv.workflow_object.value

    intro_tab, uploader_tab, detect_tab, explain_tab, examples_tab = st.tabs(
        [
            "Detect Case Patterns workflow:",
            "Prepare case data",
            "Detect attribute patterns",
            "Generate AI pattern reports",
            "View example outputs"
        ]
    )
    selected_pattern = ""
    graph_df = None
    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content)
        add_download_pdf(
            f"{workflow}_introduction_tutorial.pdf",
            file_content,
            ":floppy_disk: Download as PDF",
        )
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            ui_components.single_csv_uploader(
                "detect_case_patterns",
                "Upload CSV",
                sv.detect_case_patterns_last_file_name,
                sv.detect_case_patterns_input_df,
                sv.detect_case_patterns_final_df,
                key="case_patterns_uploader",
                height=500,
            )
        with model_col:
            ui_components.prepare_input_df(
                "detect_case_patterns",
                sv.detect_case_patterns_input_df,
                sv.detect_case_patterns_final_df
            )
            options = [""] + [
                c
                for c in sv.detect_case_patterns_final_df.value.columns.to_numpy()
                if c != "Subject ID"
            ]
            sv.detect_case_patterns_time_col.value = st.selectbox(
                "Period column",
                options,
                index=options.index(sv.detect_case_patterns_time_col.value)
                if sv.detect_case_patterns_time_col.value in options
                else 0,
            )
            time_col = sv.detect_case_patterns_time_col.value
            att_cols = [
                col
                for col in sv.detect_case_patterns_final_df.value.columns.to_numpy()
                if col not in ["Subject ID", time_col]
                and st.session_state[f"detect_case_patterns_{col}"] is True
            ]

            ready = len(att_cols) > 0 and sv.detect_case_patterns_time_col.value != ""

            if st.button("Generate attribute model", disabled=not ready):
                with st.spinner("Adding links to model..."):
                    time_col = sv.detect_case_patterns_time_col.value
                    graph_df = sv.detect_case_patterns_final_df.value.copy(deep=True)
                    dcp.generate_graph_model(
                        graph_df,
                        time_col,
                        config.type_val_sep,
                        config.min_edge_weight,
                        config.missing_edge_prop
                    )
            if ready and len(dcp.dynamic_graph_df) > 0:
                st.success(
                    f'Attribute model has **{len(dcp.dynamic_graph_df)}** links spanning **{len(dcp.dynamic_graph_df["Subject ID"].unique())}** cases, **{len(dcp.dynamic_graph_df["Full Attribute"].unique())}** attributes, and **{len(dcp.dynamic_graph_df["Period"].unique())}** periods.'
                )

    with detect_tab:
        if not ready or len(sv.detect_case_patterns_final_df.value) == 0:
            st.warning("Generate an attribute model to continue.")
        else:
            b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 2])
            with b1:
                minimum_pattern_count = st.number_input(
                    "Minimum pattern count",
                    min_value=1,
                    step=1,
                    value=sv.detect_case_patterns_min_pattern_count.value,
                    help="The minimum number of times a pattern must occur in a given period to be detectable.",
                )
            with b2:
                maximum_pattern_count = st.number_input(
                    "Maximum pattern length",
                    min_value=1,
                    step=1,
                    value=sv.detect_case_patterns_max_pattern_length.value,
                    help="The maximum number of attributes in a pattern. Longer lengths will take longer to detect.",
                )
            with b3:
                if st.button("Detect patterns"):
                    progress_text = "Starting..."
                    progress_bar = st.progress(0, text=progress_text)
                    sv.detect_case_patterns_min_pattern_count.value = (
                        minimum_pattern_count
                    )
                    sv.detect_case_patterns_max_pattern_length.value = (
                        maximum_pattern_count
                    )
                    sv.detect_case_patterns_selected_pattern.value = ""
                    sv.detect_case_patterns_selected_pattern_period.value = ""

                    with st.spinner("Processing..."):
                        sv.detect_case_patterns_table_index.value += 1
                        progress_bar.progress(25, text="Embedding graph...")
                        dcp.generate_embedding_model()
                        progress_bar.progress(50, text="Detecting data patterns...")
                        dcp.detect_patterns(
                            min_pattern_count=sv.detect_case_patterns_min_pattern_count.value,
                            max_pattern_length=sv.detect_case_patterns_max_pattern_length.value
                        )
                        progress_bar.progress(75, text="Creating time series...")
                        dcp.create_time_series_df()
                        progress_bar.progress(99, text="Finalizing...")
                        progress_bar.empty()
                        st.rerun()
            with b4:
                st.download_button(
                    "Download patterns",
                    data=dcp.patterns_df.to_csv(index=False),
                    file_name="detect_case_patterns.csv",
                    mime="text/csv",
                    disabled=len(dcp.patterns_df) == 0,
                )
            if len(dcp.patterns_df) > 0:
                period_count = len(
                    dcp.patterns_df["period"].unique()
                )
                pattern_count = len(dcp.patterns_df)
                unique_count = len(
                    dcp.patterns_df["pattern"].unique()
                )
                st.success(
                    f"Over **{period_count}** periods, detected **{pattern_count}** attribute patterns (**{unique_count}** unique) from **{dcp.close_pairs}**/**{dcp.all_pairs}** converging attribute pairs (**{round(dcp.close_pairs / dcp.all_pairs * 100, 2) if dcp.all_pairs > 0 else 0}%**). Patterns ranked by ```overall_score = normalize(length * ln(count) * z_score * detections)```."
                )
                gb = GridOptionsBuilder.from_dataframe(dcp.patterns_df)
                gb.configure_default_column(
                    flex=1,
                    wrapText=True,
                    wrapHeaderText=True,
                    enablePivot=False,
                    enableValue=False,
                    enableRowGroup=False,
                )
                gb.configure_selection(selection_mode="single", use_checkbox=False)
                gb.configure_side_bar()
                gridoptions = gb.build()
                gridoptions["columnDefs"][0]["minWidth"] = 100
                gridoptions["columnDefs"][1]["minWidth"] = 400

                st.warning(
                    "Select column headers to rank patterns by that attribute. Use quickfilter or column filters to narrow down the list of patterns. Select a pattern to continue."
                )

                response = AgGrid(
                    dcp.patterns_df,
                    key=f"report_grid_{sv.detect_case_patterns_table_index.value}",
                    height=380,
                    gridOptions=gridoptions,
                    enable_enterprise_modules=False,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=False,
                    header_checkbox_selection_filtered_only=False,
                    use_checkbox=False,
                    enable_quicksearch=True,
                    reload_data=False,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                )  # type: ignore

                selected_pattern = (
                    response["selected_rows"][0]["pattern"]
                    if len(response["selected_rows"]) > 0
                    else sv.detect_case_patterns_selected_pattern.value
                )
                print(f"selected_pattern: {selected_pattern}")
                selected_pattern_period = (
                    response["selected_rows"][0]["period"]
                    if len(response["selected_rows"]) > 0
                    else sv.detect_case_patterns_selected_pattern_period.value
                )

                if selected_pattern != "":
                    if (
                        selected_pattern
                        != sv.detect_case_patterns_selected_pattern.value
                    ):
                        sv.detect_case_patterns_selected_pattern.value = (
                            selected_pattern
                        )
                        sv.detect_case_patterns_selected_pattern_period.value = (
                            selected_pattern_period
                        )
                        sv.detect_case_patterns_report.value = ""
                        sv.detect_case_patterns_report_validation.value = {}

                        tdf = dcp.time_series_df
                        tdf = tdf[tdf["pattern"] == selected_pattern]
                        sv.detect_case_patterns_selected_pattern_df.value = tdf
                        
                    count_ct = dcp.create_time_series_chart(
                        selected_pattern,
                        selected_pattern_period
                    )
                    st.altair_chart(count_ct, use_container_width=True)                       

            elif sv.detect_case_patterns_table_index.value > 0:
                st.info("No patterns detected.")
    with explain_tab:
        if (
            not ready
            or len(sv.detect_case_patterns_final_df.value) == 0
            or sv.detect_case_patterns_selected_pattern.value == ""
        ):
            st.warning("Select a pattern to continue.")
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                sv.detect_case_patterns_selected_pattern_att_counts.value = (
                    dcp.compute_attribute_counts(
                        selected_pattern,
                        selected_pattern_period,
                    )
                )
                variables = {
                    "pattern": sv.detect_case_patterns_selected_pattern.value,
                    "period": sv.detect_case_patterns_selected_pattern_period.value,
                    "time_series": sv.detect_case_patterns_selected_pattern_df.value.to_csv(
                        index=False
                    ),
                    "attribute_counts": sv.detect_case_patterns_selected_pattern_att_counts.value.to_csv(
                        index=False
                    ),
                }

                generate, messages, reset = ui_components.generative_ai_component(
                    sv.detect_case_patterns_system_prompt, variables
                )
                if reset:
                    sv.detect_case_patterns_system_prompt.value["user_prompt"] = (
                        prompts.user_prompt
                    )
                    st.rerun()
            with c2:
                st.markdown("##### Selected attribute pattern")
                if sv.detect_case_patterns_selected_pattern.value != "":
                    count_ct = dcp.create_time_series_chart(
                        sv.detect_case_patterns_selected_pattern.value,
                        sv.detect_case_patterns_selected_pattern_period.value
                    )
                    st.altair_chart(count_ct, use_container_width=True)
                report_placeholder = st.empty()
                gen_placeholder = st.empty()

                if generate:
                    on_callback = ui_components.create_markdown_callback(
                        report_placeholder
                    )
                    connection_bar = st.progress(10, text="Connecting to AI...")

                    def empty_connection_bar():
                        def on(_) -> None:
                            connection_bar.empty()

                        on_callback = LLMCallback()
                        on_callback.on_llm_new_token = on
                        return on_callback

                    try:
                        result = ui_components.generate_text(
                            messages, callbacks=[on_callback, empty_connection_bar()]
                        )

                        sv.detect_case_patterns_report.value = result

                        # validation, messages_to_llm = ui_components.validate_ai_report(
                        #     messages, result
                        # )
                        # sv.detect_case_patterns_report_validation.value = validation
                        # sv.detect_case_patterns_report_validation_messages.value = (
                        #     messages_to_llm
                        # )
                        st.rerun()
                    except Exception as _e:
                        empty_connection_bar()
                        raise
                else:
                    if sv.detect_case_patterns_report.value == "":
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the selected attribute pattern."
                        )

                report_data = sv.detect_case_patterns_report.value
                report_placeholder.markdown(report_data)

                ui_components.report_download_ui(
                    sv.detect_case_patterns_report, "pattern_report"
                )

                # ui_components.build_validation_ui(
                #     sv.detect_case_patterns_report_validation.value,
                #     sv.detect_case_patterns_report_validation_messages.value,
                #     report_data,
                #     workflow,
                # )
    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)