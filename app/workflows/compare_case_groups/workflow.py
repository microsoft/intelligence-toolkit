# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import polars as pl
import streamlit as st

import app.util.example_outputs_ui as example_outputs_ui
import app.workflows.compare_case_groups.variables as gn_variables
from app.util import ui_components
from app.util.download_pdf import add_download_pdf
from intelligence_toolkit.compare_case_groups.api import CompareCaseGroups, prompts
from intelligence_toolkit.helpers.df_functions import fix_null_ints


def get_intro() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: gn_variables.SessionVariables, workflow=None):
    ui_components.check_ai_configuration()

    intro_tab, prepare_tab, summarize_tab, generate_tab, examples_tab = st.tabs(
        [
            "Compare case groups workflow:",
            "Prepare case data",
            "Specify group comparisons",
            "Generate AI group reports",
            "View example outputs"
        ]
    )
    ccg: CompareCaseGroups = CompareCaseGroups()

    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content)
        add_download_pdf(
            f"{workflow}_introduction_tutorial.pdf",
            file_content,
            ":floppy_disk: Download as PDF",
        )
    with prepare_tab:
        uploader_col, model_col = st.columns([1, 1])
        with uploader_col:
            ui_components.single_csv_uploader(
                workflow,
                "Upload CSV to compare",
                sv.case_groups_last_file_name,
                sv.case_groups_input_df,
                sv.case_groups_final_df,
                key="group_comparison_uploader",
                height=400,
            )
        with model_col:
            ui_components.prepare_input_df(
                workflow,
                sv.case_groups_input_df,
                sv.case_groups_final_df,
            )
            sv.case_groups_final_df.value = fix_null_ints(sv.case_groups_final_df.value)
    with summarize_tab:
        if len(sv.case_groups_final_df.value) == 0:
            st.warning("Upload data to continue.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("##### Define summary model")
                sorted_cols = sorted(sv.case_groups_final_df.value.columns)

                default_groups_existing = [
                    attr for attr in sv.case_groups_groups.value if attr in sorted_cols
                ]
                default_atts_existing = [
                    attr
                    for attr in sv.case_groups_aggregates.value
                    if attr in sorted_cols
                ]

                case_group_options = ccg.get_filter_options(
                    pl.from_pandas(sv.case_groups_final_df.value)
                )
                default_filters_existing = [
                    attr
                    for attr in sv.case_groups_filters.value
                    if attr in case_group_options
                ]

                groups = st.multiselect(
                    "Compare groups of records with different combinations of these attributes:",
                    sorted_cols,
                    default=sv.case_groups_groups.value
                    if (
                        len(default_groups_existing) == len(sv.case_groups_groups.value)
                    )
                    else [],
                )
                aggregates = st.multiselect(
                    "Using counts of these attributes:",
                    sorted_cols,
                    default=sv.case_groups_aggregates.value
                    if (len(default_atts_existing) == len(sv.case_groups_groups.value))
                    else [],
                )
                temporal_options = ["", *sorted_cols]
                temporal = st.selectbox(
                    "Across windows of this temporal/ordinal attribute (optional):",
                    temporal_options,
                    index=temporal_options.index(sv.case_groups_temporal.value)
                    if sv.case_groups_temporal.value in temporal_options
                    else 0,
                )
                filters = st.multiselect(
                    "After filtering to records matching these values (optional):",
                    case_group_options,
                    default=sv.case_groups_filters.value
                    if (
                        len(default_filters_existing)
                        == len(sv.case_groups_filters.value)
                    )
                    else [],
                )

                create = st.button(
                    "Create data summary", disabled=len(groups) == 0 or len(aggregates) == 0
                )

            with c2:
                st.markdown("##### Data summary")
                if create:
                    sv.case_groups_filters.value = filters
                    sv.case_groups_groups.value = groups
                    sv.case_groups_aggregates.value = aggregates
                    sv.case_groups_temporal.value = temporal

                    ccg.create_data_summary(
                        pl.from_pandas(sv.case_groups_final_df.value),
                        filters,
                        groups,
                        aggregates,
                        temporal,
                    )
                    sv.case_groups_description.value = ccg.get_summary_description()
                    sv.case_groups_model_df.value = ccg.model_df.to_pandas()
                    st.rerun()
                if len(sv.case_groups_model_df.value) > 0:
                    st.dataframe(
                        sv.case_groups_model_df.value,
                        hide_index=True,
                        use_container_width=True,
                        height=500,
                    )

                    st.markdown(sv.case_groups_description.value)
                    st.download_button(
                        "Download data summary",
                        data=sv.case_groups_model_df.value.to_csv(
                            index=False, encoding="utf-8-sig"
                        ),
                        file_name="group_data_summary.csv",
                        mime="text/csv",
                    )

    with generate_tab:
        if len(sv.case_groups_model_df.value) == 0:
            st.warning("Prepare data summary to continue.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### Data summary filters")
                b1, b2 = st.columns([1, 1])

                ccg.model_df = pl.from_pandas(sv.case_groups_model_df.value)
                ccg.groups = sv.case_groups_groups.value
                with b1:
                    selected_groups = st.multiselect(
                        "Select specific groups to report on:",
                        ccg.get_report_groups_filter_options(),
                        format_func=lambda group_dict: ui_components.format_report_group_options(
                            group_dict, sv.case_groups_groups.value
                        ),
                        default=sv.case_groups_selected_groups.value,
                    )
                with b2:
                    top_group_ranks = st.number_input(
                        "OR Select top group ranks to report on:",
                        min_value=0,
                        max_value=9999999999,
                        value=sv.case_groups_top_groups.value,
                    )
                fdf = sv.case_groups_model_df.value.copy(deep=True)

                report_data, filter_description = ccg.get_report_data(
                    selected_groups if len(selected_groups) > 0 else None,
                    top_group_ranks if top_group_ranks > 0 else None,
                )
                num_rows = len(report_data)
                st.markdown(f"##### Filtered data summary to report on ({num_rows} rows)")
                st.dataframe(fdf, hide_index=True, use_container_width=True, height=280)
                variables = {
                    "description": sv.case_groups_description.value,
                    "dataset": report_data.to_pandas().to_csv(
                        index=False, encoding="utf-8-sig"
                    ),
                    "filters": filter_description,
                }

                generate, messages, reset = ui_components.generative_ai_component(
                    sv.case_groups_system_prompt, variables
                )
                if reset:
                    sv.case_groups_system_prompt.value["user_prompt"] = (
                        prompts.user_prompt
                    )
                    st.rerun()
            with c2:
                st.markdown("##### Group comparison report")

                narrative_placeholder = st.empty()
                gen_placeholder = st.empty()
                if generate:
                    sv.case_groups_selected_groups.value = selected_groups
                    sv.case_groups_top_groups.value = top_group_ranks

                    on_callback = ui_components.create_markdown_callback(
                        narrative_placeholder
                    )
                    result = ui_components.generate_text(
                        messages, callbacks=[on_callback]
                    )
                    sv.case_groups_report.value = result

                    # validation, messages_to_llm = ui_components.validate_ai_report(
                    #     messages, result
                    # )
                    # sv.case_groups_report_validation.value = validation
                    # sv.case_groups_report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if sv.case_groups_report.value == "":
                        gen_placeholder.warning(
                            "Press the Generate button to create an AI report for the selected groups."
                        )
                narrative_placeholder.markdown(sv.case_groups_report.value)

                ui_components.report_download_ui(sv.case_groups_report, "group_report")

                # ui_components.build_validation_ui(
                #     sv.case_groups_report_validation.value,
                #     sv.case_groups_report_validation_messages.value,
                #     sv.case_groups_report.value,
                #     workflow,
                # )
    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)

