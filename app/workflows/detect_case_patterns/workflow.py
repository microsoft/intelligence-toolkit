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

import app.workflows.detect_case_patterns.variables as ap_variables
from app.util import ui_components
from toolkit.AI.classes import LLMCallback
from toolkit.detect_case_patterns import prompts
from toolkit.detect_case_patterns.config import (
    correlation,
    diaga,
    laplacian,
    min_edge_weight,
    missing_edge_prop,
    type_val_sep,
)
from toolkit.detect_case_patterns.model import (
    compute_attribute_counts,
    create_time_series_df,
    detect_patterns,
    generate_graph_model,
    prepare_graph,
)
from toolkit.detect_case_patterns.record_counter import RecordCounter
from toolkit.graph.graph_fusion_encoder_embedding import (
    generate_graph_fusion_encoder_embedding,
)


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: ap_variables.SessionVariables, workflow):
    intro_tab, uploader_tab, detect_tab, explain_tab, examples_tab = st.tabs(
        [
            "Detect case patterns workflow:",
            "Prepare case data",
            "Detect attribute patterns",
            "Generate AI pattern reports",
            "View example outputs"
        ]
    )
    selected_pattern = ""
    graph_df = None
    with intro_tab:
        st.markdown(get_intro())
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            ui_components.single_csv_uploader(
                "detect_case_patterns",
                "Upload CSV",
                sv.detect_case_patterns_last_file_name,
                sv.detect_case_patterns_input_df,
                sv.detect_case_patterns_final_df,
                sv.detect_case_patterns_upload_key.value,
                key="case_patterns_uploader",
                height=500,
            )
        with model_col:
            ui_components.prepare_input_df(
                "detect_case_patterns",
                sv.detect_case_patterns_input_df,
                sv.detect_case_patterns_final_df
            )
            sv.detect_case_patterns_final_df.value['Subject ID'] = range(len(sv.detect_case_patterns_final_df.value))
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
                    pdf = generate_graph_model(graph_df, time_col, type_val_sep)
                sv.detect_case_patterns_dynamic_df.value = pdf
            if ready and len(sv.detect_case_patterns_dynamic_df.value) > 0:
                st.success(
                    f'Attribute model has **{len(sv.detect_case_patterns_dynamic_df.value)}** links spanning **{len(sv.detect_case_patterns_dynamic_df.value["Subject ID"].unique())}** cases, **{len(sv.detect_case_patterns_dynamic_df.value["Full Attribute"].unique())}** attributes, and **{len(sv.detect_case_patterns_dynamic_df.value["Period"].unique())}** periods.'
                )

    with detect_tab:
        if not ready or len(sv.detect_case_patterns_final_df.value) == 0:
            st.warning("Generate a graph model to continue.")
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
                        progress_bar.progress(20, text="Preparing graph...")

                        sv.detect_case_patterns_df.value, period_to_graph = (
                            prepare_graph(
                                sv.detect_case_patterns_dynamic_df.value,
                                min_edge_weight,
                                missing_edge_prop,
                            )
                        )
                        node_to_label_str = dict(
                            sv.detect_case_patterns_dynamic_df.value[
                                ["Full Attribute", "Attribute Type"]
                            ].values
                        )
                        # convert string labels to int labels
                        sorted_labels = sorted(set(node_to_label_str.values()))
                        label_to_code = {v: i for i, v in enumerate(sorted_labels)}
                        node_to_label = {
                            k: label_to_code[v] for k, v in node_to_label_str.items()
                        }
                        progress_bar.progress(40, text="Generating embedding...")
                        (sv.detect_case_patterns_node_to_period_to_pos.value, _) = (
                            generate_graph_fusion_encoder_embedding(
                                period_to_graph,
                                node_to_label,
                                correlation,
                                diaga,
                                laplacian,
                            )
                        )

                        sv.detect_case_patterns_record_counter.value = RecordCounter(
                            sv.detect_case_patterns_dynamic_df.value
                        )
                        progress_bar.progress(60, text="Detecting data patterns...")
                        (
                            sv.detect_case_patterns_pattern_df.value,
                            sv.detect_case_patterns_close_pairs.value,
                            sv.detect_case_patterns_all_pairs.value,
                        ) = detect_patterns(
                            sv.detect_case_patterns_node_to_period_to_pos.value,
                            sv.detect_case_patterns_dynamic_df.value,
                            type_val_sep,
                            sv.detect_case_patterns_min_pattern_count.value,
                            sv.detect_case_patterns_max_pattern_length.value,
                        )
                        progress_bar.progress(80, text="Creating time series...")

                        tdf = create_time_series_df(
                            sv.detect_case_patterns_dynamic_df.value,
                            sv.detect_case_patterns_pattern_df.value,
                        )
                        progress_bar.progress(99, text="Finalizing...")
                        sv.detect_case_patterns_time_series_df.value = tdf
                        progress_bar.empty()
                        st.rerun()
            with b4:
                st.download_button(
                    "Download patterns",
                    data=sv.detect_case_patterns_pattern_df.value.to_csv(index=False),
                    file_name="detect_case_patterns.csv",
                    mime="text/csv",
                    disabled=len(sv.detect_case_patterns_pattern_df.value) == 0,
                )
            if len(sv.detect_case_patterns_pattern_df.value) > 0:
                period_count = len(
                    sv.detect_case_patterns_pattern_df.value["period"].unique()
                )
                pattern_count = len(sv.detect_case_patterns_pattern_df.value)
                unique_count = len(
                    sv.detect_case_patterns_pattern_df.value["pattern"].unique()
                )
                st.success(
                    f"Over **{period_count}** periods, detected **{pattern_count}** attribute patterns (**{unique_count}** unique) from **{sv.detect_case_patterns_close_pairs.value}**/**{sv.detect_case_patterns_all_pairs.value}** converging attribute pairs (**{round(sv.detect_case_patterns_close_pairs.value / sv.detect_case_patterns_all_pairs.value * 100, 2) if sv.detect_case_patterns_all_pairs.value > 0 else 0}%**). Patterns ranked by ```overall_score = normalize(length * ln(count) * z_score * detections)```."
                )
                show_df = sv.detect_case_patterns_pattern_df.value

                gb = GridOptionsBuilder.from_dataframe(show_df)
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
                response = AgGrid(
                    show_df,
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

                        tdf = sv.detect_case_patterns_time_series_df.value
                        tdf = tdf[tdf["pattern"] == selected_pattern]
                        sv.detect_case_patterns_selected_pattern_df.value = tdf
                        sv.detect_case_patterns_selected_pattern_att_counts.value = (
                            compute_attribute_counts(
                                sv.detect_case_patterns_final_df.value,
                                selected_pattern,
                                time_col,
                                selected_pattern_period,
                                type_val_sep,
                            )
                        )
                    title = 'Pattern: ' + selected_pattern + ' (' + selected_pattern_period + ')'
                    count_ct = (
                        alt.Chart(sv.detect_case_patterns_selected_pattern_df.value)
                        .mark_line()
                        .encode(x="period:O", y="count:Q", color=alt.ColorValue("blue"))
                        .properties(title=title,
                                    height=220, width=600)
                    )
                    st.altair_chart(count_ct, use_container_width=True)                       
                    st.warning(
                        "Select column headers to rank patterns by that attribute. Use quickfilter or column filters to narrow down the list of patterns. Select a pattern to continue."
                    )
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
                    tdf = sv.detect_case_patterns_selected_pattern_df.value
                    title = 'Pattern: ' + selected_pattern + ' (' + selected_pattern_period + ')'
                    count_ct = (
                        alt.Chart(tdf)
                        .mark_line()
                        .encode(x="period:O", y="count:Q", color=alt.ColorValue("blue"))
                        .properties(title=title, height=220, width=600)
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

                        validation, messages_to_llm = ui_components.validate_ai_report(
                            messages, result
                        )
                        sv.detect_case_patterns_report_validation.value = validation
                        sv.detect_case_patterns_report_validation_messages.value = (
                            messages_to_llm
                        )
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

                ui_components.build_validation_ui(
                    sv.detect_case_patterns_report_validation.value,
                    sv.detect_case_patterns_report_validation_messages.value,
                    report_data,
                    workflow,
                )
    with examples_tab:

        workflow_home = 'example_outputs/detect_case_patterns'

        mock_data_folders = [x for x in os.listdir(f'{workflow_home}')]
        print(mock_data_folders)
        selected_data = st.selectbox('Select example', mock_data_folders)
        if selected_data != None:
            t1, t2, t3, t4 = st.tabs(['Input data', 'Prepared data', 'Case patterns', 'Pattern reports'])
            with t1:
                data_file = f'{workflow_home}/{selected_data}/{selected_data}_input.csv'
                df = pd.read_csv(data_file)
                st.dataframe(df, height=500, use_container_width=True)
                st.download_button(
                    label=f'Download {selected_data}_input.csv',
                    data=df.to_csv(index=False, encoding='utf-8'),
                    file_name=data_file,
                    mime='text/csv',
                )
            with t2:
                data_file = f'{workflow_home}/{selected_data}/{selected_data}_prepared.csv'
                df = pd.read_csv(data_file)
                st.dataframe(df, height=500, use_container_width=True)
                st.download_button(
                    label=f'Download {selected_data}_prepared.csv',
                    data=df.to_csv(index=False, encoding='utf-8'),
                    file_name=data_file,
                    mime='text/csv',
                )
            with t3:
                data_file = f'{workflow_home}/{selected_data}/{selected_data}_case_patterns.csv'
                df = pd.read_csv(data_file)
                st.dataframe(df, height=500, use_container_width=True)
                st.download_button(
                    label=f'Download {selected_data}_case_patterns.csv',
                    data=df.to_csv(index=False, encoding='utf-8'),
                    file_name=data_file,
                    mime='text/csv',
                )
            with t4:
                pattern_md = []
                pattern_img = []
                index = 1
                while True:
                    if os.path.exists(f'{workflow_home}/{selected_data}/{selected_data}_pattern_report_{index}.md') and \
                        os.path.exists(f'{workflow_home}/{selected_data}/{selected_data}_pattern_chart_{index}.png'):
                        with open(f'{workflow_home}/{selected_data}/{selected_data}_pattern_report_{index}.md', 'r') as f:
                            pattern_md.append(f.read())
                        with open(f'{workflow_home}/{selected_data}/{selected_data}_pattern_chart_{index}.png', 'rb') as f:
                            im = PIL.Image.open(io.BytesIO(f.read()))
                            pattern_img.append(im)
                        index += 1
                    else:
                        break
                for i in range(len(pattern_md)):
                    st.divider()
                    st.markdown(f'# Example {i+1}')
                    st.divider()
                    st.image(pattern_img[i])
                    st.markdown(pattern_md[i])
                    st.divider()
                    