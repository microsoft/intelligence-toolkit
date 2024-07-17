# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import altair as alt
import streamlit as st
import workflows.attribute_patterns.variables as ap_variables
from st_aggrid import (
    AgGrid,
    ColumnsAutoSizeMode,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)
from util import ui_components

from python.attribute_patterns import prompts
from python.attribute_patterns.embedding import generate_embedding
from python.attribute_patterns.model import (
    compute_attribute_counts,
    create_time_series_df,
    detect_patterns,
    generate_graph_model,
    prepare_graph,
)
from python.attribute_patterns.record_counter import RecordCounter


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: ap_variables.SessionVariables, workflow):
    intro_tab, uploader_tab, detect_tab, explain_tab = st.tabs([
        "Attribute patterns workflow:",
        "Create graph model",
        "Detect patterns",
        "Generate AI pattern reports",
    ])
    graph_df = None
    with intro_tab:
        st.markdown(get_intro())
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            ui_components.single_csv_uploader(
                "attribute_patterns",
                "Upload CSV",
                sv.attribute_last_file_name,
                sv.attribute_input_df,
                sv.attribute_binned_df,
                sv.attribute_final_df,
                sv.attribute_upload_key.value,
                key="attributes_uploader",
                height=500,
            )
        with model_col:
            ui_components.prepare_input_df(
                "attribute_patterns",
                sv.attribute_input_df,
                sv.attribute_binned_df,
                sv.attribute_final_df,
                sv.attribute_subject_identifier,
            )
            options = [""] + [
                c
                for c in sv.attribute_final_df.value.columns.to_numpy()
                if c != "Subject ID"
            ]
            sv.attribute_time_col.value = st.selectbox(
                "Period column",
                options,
                index=options.index(sv.attribute_time_col.value)
                if sv.attribute_time_col.value in options
                else 0,
            )
            time_col = sv.attribute_time_col.value
            att_cols = [
                col
                for col in sv.attribute_final_df.value.columns.to_numpy()
                if col not in ["Subject ID", time_col]
                and st.session_state[f"attribute_patterns_{col}"] is True
            ]

            ready = len(att_cols) > 0 and sv.attribute_time_col.value != ""

            if st.button("Generate graph model", disabled=not ready):
                with st.spinner("Adding links to model..."):
                    time_col = sv.attribute_time_col.value
                    graph_df = sv.attribute_final_df.value.copy(deep=True)
                    pdf = generate_graph_model(graph_df, time_col)
                sv.attribute_dynamic_df.value = pdf
            if ready and len(sv.attribute_dynamic_df.value) > 0:
                st.success(
                    f'Graph model has **{len(sv.attribute_dynamic_df.value)}** links spanning **{len(sv.attribute_dynamic_df.value["Subject ID"].unique())}** cases, **{len(sv.attribute_dynamic_df.value["Full Attribute"].unique())}** attributes, and **{len(sv.attribute_dynamic_df.value["Period"].unique())}** periods.'
                )

    with detect_tab:
        if not ready or len(sv.attribute_final_df.value) == 0:
            st.warning("Generate a graph model to continue.")
        else:
            b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 2])
            with b1:
                minimum_pattern_count = st.number_input(
                    "Minimum pattern count",
                    min_value=1,
                    step=1,
                    value=sv.attribute_min_pattern_count.value,
                    help="The minimum number of times a pattern must occur in a given period to be detectable.",
                )
            with b2:
                maximum_pattern_count = st.number_input(
                    "Maximum pattern length",
                    min_value=1,
                    step=1,
                    value=sv.attribute_max_pattern_length.value,
                    help="The maximum number of attributes in a pattern. Longer lengths will take longer to detect.",
                )
            with b3:
                if st.button("Detect patterns"):
                    sv.attribute_min_pattern_count.value = minimum_pattern_count
                    sv.attribute_max_pattern_length.value = maximum_pattern_count
                    sv.attribute_selected_pattern.value = ""
                    sv.attribute_selected_pattern_period.value = ""

                    with st.spinner("Detecting patterns..."):
                        sv.attribute_table_index.value += 1
                        sv.attribute_df.value, time_to_graph = prepare_graph(
                            sv.attribute_dynamic_df.value
                        )
                        (
                            sv.attribute_embedding_df.value,
                            sv.attribute_node_to_centroid.value,
                            sv.attribute_period_embeddings.value,
                        ) = generate_embedding(sv.attribute_df.value, time_to_graph)
                        sv.attribute_record_counter.value = RecordCounter(
                            sv.attribute_dynamic_df.value
                        )

                        (
                            sv.attribute_pattern_df.value,
                            sv.attribute_close_pairs.value,
                            sv.attribute_all_pairs.value,
                        ) = detect_patterns(
                            sv.attribute_node_to_centroid.value,
                            sv.attribute_period_embeddings.value,
                            sv.attribute_dynamic_df.value,
                            sv.attribute_min_pattern_count.value,
                            sv.attribute_max_pattern_length.value,
                        )
                        st.rerun()
            with b4:
                st.download_button(
                    "Download patterns",
                    data=sv.attribute_pattern_df.value.to_csv(index=False),
                    file_name="attribute_patterns.csv",
                    mime="text/csv",
                    disabled=len(sv.attribute_pattern_df.value) == 0,
                )
            if len(sv.attribute_pattern_df.value) > 0:
                period_count = len(sv.attribute_pattern_df.value["period"].unique())
                pattern_count = len(sv.attribute_pattern_df.value)
                unique_count = len(sv.attribute_pattern_df.value["pattern"].unique())
                st.success(
                    f"Over **{period_count}** periods, detected **{pattern_count}** attribute patterns (**{unique_count}** unique) from **{sv.attribute_close_pairs.value}**/**{sv.attribute_all_pairs.value}** converging attribute pairs (**{round(sv.attribute_close_pairs.value / sv.attribute_all_pairs.value * 100, 2) if sv.attribute_all_pairs.value > 0 else 0}%**). Patterns ranked by ```overall_score = normalize(length * ln(count) * z_score * detections)```."
                )
                show_df = sv.attribute_pattern_df.value
                tdf = create_time_series_df(
                    sv.attribute_dynamic_df.value, sv.attribute_pattern_df.value
                )
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
                    key=f"report_grid_{sv.attribute_table_index.value}",
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
                    else sv.attribute_selected_pattern.value
                )
                selected_pattern_period = (
                    response["selected_rows"][0]["period"]
                    if len(response["selected_rows"]) > 0
                    else sv.attribute_selected_pattern_period.value
                )

                if selected_pattern != "":
                    if selected_pattern != sv.attribute_selected_pattern.value:
                        sv.attribute_selected_pattern.value = selected_pattern
                        sv.attribute_selected_pattern_period.value = (
                            selected_pattern_period
                        )
                        sv.attribute_report.value = ""
                        sv.attribute_report_validation.value = {}
                        st.rerun()

                    st.markdown(
                        "**Selected pattern: "
                        + selected_pattern
                        + " ("
                        + selected_pattern_period
                        + ")**"
                    )
                    tdf = tdf[tdf["pattern"] == selected_pattern]
                    sv.attribute_selected_pattern_df.value = tdf
                    sv.attribute_selected_pattern_att_counts.value = (
                        compute_attribute_counts(
                            sv.attribute_final_df.value,
                            selected_pattern,
                            time_col,
                            selected_pattern_period,
                        )
                    )
                    count_ct = (
                        alt.Chart(tdf)
                        .mark_line()
                        .encode(x="period:O", y="count:Q", color=alt.ColorValue("blue"))
                        .properties(height=200, width=600)
                    )
                    st.altair_chart(count_ct, use_container_width=True)
                else:
                    st.warning(
                        "Select column headers to rank patterns by that attribute. Use quickfilter or column filters to narrow down the list of patterns. Select a pattern to continue."
                    )
            elif sv.attribute_table_index.value > 0:
                st.info("No patterns detected.")
    with explain_tab:
        if (
            not ready
            or len(sv.attribute_final_df.value) == 0
            or sv.attribute_selected_pattern.value == ""
        ):
            st.warning("Select a pattern to continue.")
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                variables = {
                    "pattern": sv.attribute_selected_pattern.value,
                    "period": sv.attribute_selected_pattern_period.value,
                    "time_series": sv.attribute_selected_pattern_df.value.to_csv(
                        index=False
                    ),
                    "attribute_counts": sv.attribute_selected_pattern_att_counts.value.to_csv(
                        index=False
                    ),
                }

                generate, messages, reset = ui_components.generative_ai_component(
                    sv.attribute_system_prompt, variables
                )
                if reset:
                    sv.attribute_system_prompt.value["user_prompt"] = (
                        prompts.user_prompt
                    )
                    st.rerun()
            with c2:
                st.markdown("##### Selected attribute pattern")
                if sv.attribute_selected_pattern.value != "":
                    st.markdown(
                        "**"
                        + sv.attribute_selected_pattern.value
                        + " ("
                        + sv.attribute_selected_pattern_period.value
                        + ")**"
                    )
                    tdf = sv.attribute_selected_pattern_df.value

                    count_ct = (
                        alt.Chart(tdf)
                        .mark_line()
                        .encode(x="period:O", y="count:Q", color=alt.ColorValue("blue"))
                        .properties(height=200, width=600)
                    )
                    st.altair_chart(count_ct, use_container_width=True)
                report_placeholder = st.empty()
                gen_placeholder = st.empty()

                if generate:
                    on_callback = ui_components.create_markdown_callback(
                        report_placeholder
                    )
                    result = ui_components.generate_text(
                        messages, callbacks=[on_callback]
                    )
                    sv.attribute_report.value = result

                    validation, messages_to_llm = ui_components.validate_ai_report(
                        messages, result
                    )
                    sv.attribute_report_validation.value = validation
                    sv.attribute_report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if sv.attribute_report.value == "":
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the selected attribute pattern."
                        )

                report_data = sv.attribute_report.value
                report_placeholder.markdown(report_data)

                ui_components.report_download_ui(sv.attribute_report, "pattern_report")

                ui_components.build_validation_ui(
                    sv.attribute_report_validation.value,
                    sv.attribute_report_validation_messages.value,
                    report_data,
                    workflow,
                )
