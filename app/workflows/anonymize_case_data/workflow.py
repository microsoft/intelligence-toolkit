# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import plotly.io as pio
import streamlit as st

import app.util.example_outputs_ui as example_outputs_ui
import app.util.ui_components as ui_components
import app.workflows.anonymize_case_data.config as config
import app.workflows.anonymize_case_data.variables as ds_variables
import intelligence_toolkit.anonymize_case_data.visuals as visuals
from app.util.download_pdf import add_download_pdf
from intelligence_toolkit.anonymize_case_data.api import AnonymizeCaseData
from intelligence_toolkit.anonymize_case_data.visuals import color_schemes


def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(file_path) as file:
        return file.read()


def create(sv: ds_variables.SessionVariables, workflow: None):
    intro_tab, prepare_tab, generate_tab, queries_tab, examples_tab = st.tabs(
        [
            "Anonymize Case Data workflow:",
            "Prepare sensitive data",
            "Generate anonymous data",
            "Query and visualize data",
            "View example outputs"
        ]
    )
    df = None
    acd: AnonymizeCaseData = sv.workflow_object.value
    with intro_tab:
        file_content = get_intro()
        st.markdown(file_content)
        add_download_pdf(
            f"{workflow}_introduction_tutorial.pdf",
            file_content,
            ":floppy_disk: Download as PDF",
        )
    with prepare_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            ui_components.single_csv_uploader(
                workflow,
                "Upload sensitive data CSV",
                sv.anonymize_last_sensitive_file_name,
                sv.anonymize_raw_sensitive_df,
                sv.anonymize_sensitive_df,
                key="sensitive_data_uploader",
                height=400,
            )
        with model_col:
            ui_components.prepare_input_df(
                workflow,
                sv.anonymize_raw_sensitive_df,
                sv.anonymize_sensitive_df
            )

            if len(sv.anonymize_sensitive_df.value) > 0:
                
                syn_stats = acd.analyze_synthesizability(sv.anonymize_sensitive_df.value)

                st.markdown("### Anonymizability summary")
                st.markdown(
                    f"Number of selected columns: **{syn_stats.num_cols}**",
                    help="This is the number of columns you selected for processing. The more columns you select, the harder it will be to anonymize data.",
                )
                st.markdown(
                    f"Number of distinct attribute values: **{syn_stats.overall_att_count}**",
                    help="This is the total number of distinct attribute values across all selected columns. The more distinct values, the harder it will be to anonymize data.",
                )
                st.markdown(
                    f"Theoretical attribute combinations: **{syn_stats.possible_combinations}**",
                    help="This is the total number of possible attribute combinations across all selected columns. The higher this number, the harder it will be to anonymize data.",
                )
                st.markdown(
                    f"Theoretical combinations per record: **{syn_stats.possible_combinations_per_row}**",
                    help="This is the mean number of possible attribute combinations per sensitive case record. The higher this number, the harder it will be to anonymize data.",
                )
                st.markdown(
                    f"Typical values per record: **{round(syn_stats.mean_vals_per_record, 1)}**",
                    help="This is the mean number of actual attribute values per sensitive case record. The higher this number, the harder it will be to anonymize data.",
                )
                st.markdown(
                    f"Typical combinations per record: **{round(syn_stats.max_combinations_per_record, 1)}**",
                    help="This is the number of attribute combinations in a record with the typical number of values.",
                )
                st.markdown(
                    f"**Excess combinations ratio: {round(syn_stats.excess_combinations_ratio, 1)}**",
                    help="This is the ratio of theoretical combinations per record to the typical combinations per record. The higher this number, the harder it will be to anonymize data. **Rule of thumb**: Aim for a ratio of **5** or lower.",
                )
                if syn_stats.excess_combinations_ratio <= 5:
                    st.success(
                        "This dataset is likely to be anonymizable. You can proceed to anonymize the data."
                    )
                else:
                    st.warning(
                        "This dataset may be difficult to anonymize. You may need to reduce the number of columns or attribute values to proceed."
                    )

    with generate_tab:
        if len(sv.anonymize_sensitive_df.value) == 0:
            st.warning("Please upload and prepare data to continue.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown("#### Anonymize data")
                b1, b2, b3 = st.columns([1, 1, 1])

                with b1:
                    epsilon = st.number_input(
                        "Epsilon",
                        value=sv.anonymize_epsilon.value,
                        help="The privacy budget, under differential privacy, to use when synthesizing the aggregate dataset.\n\nLower values of epsilon correspond to greater privacy protection but lower data quality.\n\nThe delta parameter is set automatically as 1/(protected_records*ln(protected_records)), where protected_records is the count of sensitive records protected using 0.5% of the privacy budget.\n\n**Rule of thumb**: Aim to keep epsilon at **12** or below.",
                    )
                with b2:
                    fab_mode = st.selectbox(
                        "Fabrication mode",
                        options=["Balanced", "Progressive", "Minimized", "Uncontrolled"],
                        help="Options for controlling the fabrication of attribute combinations in the anonymized data. Experiment with different settings and compare the resulting data quality."
                    )
                with b3:
                    if st.button("Anonymize data"):
                        sv.anonymize_epsilon.value = epsilon
                        df = sv.anonymize_sensitive_df.value
                        with st.spinner("Anonymizing data..."):
                            fab_option = AnonymizeCaseData.FabricationStrategy.BALANCED if fab_mode == "Balanced" \
                                else AnonymizeCaseData.FabricationStrategy.PROGRESSIVE if fab_mode == "Progressive" \
                                else AnonymizeCaseData.FabricationStrategy.MINIMIZED if fab_mode == "Minimized" \
                                else AnonymizeCaseData.FabricationStrategy.UNCONTROLLED
                                

                            acd.anonymize_case_data(
                                df=df,
                                epsilon=epsilon,
                                fabrication_mode=fab_option
                            )
                            sv.anonymize_synthetic_df.value = acd.synthetic_df
                            sv.anonymize_aggregate_df.value = acd.aggregate_df
                            sv.anonymize_delta.value = f"{acd.delta:.2e}"
                if epsilon > 12:
                    st.warning(
                        "Epsilon is above the recommended threshold of 12"
                    )
                st.markdown(
                    "#### Analyze data",
                    help="Tables show three evaluation metrics for each **Length** of attribute combination up to 4, plus an **Overall** average.\n\n- **Count +/- Error** is the average number of records for the combination length +/- the average absolute error in the number of records.\n- **Suppressed %** is the percentage of the total attribute counts that were suppressed, i.e., present in the Sensitive data but not the Aggregate/Synthetic data.\n- **Fabricated %** is the percentage of the total attribute counts that were fabricated, i.e., present in the Aggregate/Synthetic data but not the Sensitive data.\n\nPercentages are calculated with respect to attribute counts in the Sensitive data.\n\n**Rule of thumb**: For the Synthetic data, aim to keep the Overall Error below the Overall Count, Suppressed % below 10%, and Fabricated % below 1%.",
                )

                if len(acd.aggregate_error_report) > 0:
                    st.markdown(
                        f"Differential privacy parameters: **Epsilon = {sv.anonymize_epsilon.value}**, **Delta = {sv.anonymize_delta.value}**"
                    )
                    st.markdown("###### Aggregate data quality")
                    st.dataframe(
                        acd.aggregate_error_report,
                        hide_index=True,
                        use_container_width=False,
                    )
                    error_str = acd.aggregate_error_report[acd.aggregate_error_report["Length"] == "Overall"]["Count +/- Error"].values[0]
                    mean_count, mean_error = error_str.split(" +/- ")
                    if float(mean_error) <= float(mean_count):
                        st.success("Error < Count on average: data quality is good")
                    else:
                        st.warning("Error > Count on average: simplify sensitive data to improve")
                    st.markdown("###### Synthetic data quality")
                    st.dataframe(
                        acd.synthetic_error_report,
                        hide_index=True,
                        use_container_width=False,
                    )
                    error_str = acd.synthetic_error_report[acd.synthetic_error_report["Length"] == "Overall"]["Count +/- Error"].values[0]
                    mean_count, mean_error = error_str.split(" +/- ")
                    if float(mean_error) <= float(mean_count):
                        st.success("Error < Count on average: data quality is good")
                    else:
                        st.warning("Error > Count on average: simplify sensitive data to improve")
                    st.warning(
                        "**Caution**: These tables should only be used to evaluate the quality of data for release. Sharing them compromises privacy."
                    )

            with c2:
                st.markdown("##### Aggregate data")
                if len(sv.anonymize_aggregate_df.value) > 0:
                    st.dataframe(
                        sv.anonymize_aggregate_df.value,
                        hide_index=True,
                        use_container_width=True,
                        height=700,
                    )
                    st.download_button(
                        "Download Aggregate data",
                        data=sv.anonymize_aggregate_df.value.to_csv(index=False),
                        file_name="aggregate_data.csv",
                        mime="text/csv",
                    )

            with c3:
                st.markdown("##### Synthetic data")
                if len(sv.anonymize_synthetic_df.value) > 0:
                    st.dataframe(
                        sv.anonymize_synthetic_df.value,
                        hide_index=True,
                        use_container_width=True,
                        height=700,
                    )
                    st.download_button(
                        "Download Synthetic data",
                        data=sv.anonymize_synthetic_df.value.to_csv(index=False),
                        file_name="synthetic_data.csv",
                        mime="text/csv",
                    )

    with queries_tab:
        if (
            len(sv.anonymize_synthetic_df.value) == 0
            or len(sv.anonymize_aggregate_df.value) == 0
        ):
            st.warning(
                "Please synthesize data to continue, or upload an existing synthetic dataset below."
            )
            ui_components.single_csv_uploader(
                workflow,
                "Upload synthetic data CSV",
                sv.anonymize_last_synthetic_file_name,
                sv.anonymize_synthetic_df,
                None,
                key="synthetic_data_uploader",
            )
            ui_components.single_csv_uploader(
                workflow,
                "Upload aggregate data CSV",
                sv.anonymize_last_aggregate_file_name,
                sv.anonymize_aggregate_df,
                None,
                key="aggregate_data_uploader",
            )
            if (
                len(sv.anonymize_synthetic_df.value) > 0
                and len(sv.anonymize_aggregate_df.value) > 0
            ):
                st.rerun()
        else:
            container = st.container(border=True)
            scheme_options = sorted(color_schemes.keys())
            chart_type_options = ["Top attributes", "Time series", "Flow (alluvial)"]

            if f"{workflow}_query_selections" not in st.session_state:
                st.session_state[f"{workflow}_query_selections"] = []
            if f"{workflow}_unit" not in st.session_state:
                st.session_state[f"{workflow}_unit"] = ""
            if f"{workflow}_scheme" not in st.session_state:
                st.session_state[f"{workflow}_scheme"] = scheme_options[0]
            if f"{workflow}_chart_width" not in st.session_state:
                st.session_state[f"{workflow}_chart_width"] = 800
            if f"{workflow}_chart_height" not in st.session_state:
                st.session_state[f"{workflow}_chart_height"] = 400
            if f"{workflow}_chart_type" not in st.session_state:
                st.session_state[f"{workflow}_chart_type"] = chart_type_options[0]
            if f"{workflow}_chart_individual_configuration" not in st.session_state:
                st.session_state[f"{workflow}_chart_individual_configuration"] = {}
            if f"{workflow}_time_attributes" not in st.session_state:
                st.session_state[f"{workflow}_time_attributes"] = ""
            if f"{workflow}_series_attributes" not in st.session_state:
                st.session_state[f"{workflow}_series_attributes"] = []

            adf = sv.anonymize_aggregate_df.value
            adf["protected_count"] = adf["protected_count"].astype(int)
            sdf = sv.anonymize_synthetic_df.value.copy(deep=True)
            options = []
            for att in sdf.columns.to_numpy():
                vals = [f"{att}:{x}" for x in sdf[att].unique() if len(str(x)) > 0]
                vals.sort()
                options.extend(vals)
            c1, c2 = st.columns([1, 2])
            val_separator = ":"
            att_separator = ";"
            data_schema = acd.get_data_schema()
            with c1:
                st.markdown("##### Constuct query")
                if len(sdf) > 0:
                    count_holder = st.empty()

                    filters = st.multiselect(
                        label="Add attributes to query",
                        options=options,
                        default=st.session_state[f"{workflow}_query_selections"],
                    )

                    selection = []
                    for att, vals in data_schema.items():
                        filter_vals = [v for v in vals if f"{att}:{v}" in filters]
                        if len(filter_vals) > 0:
                            sdf = sdf[sdf[att].isin(filter_vals)]
                            for val in filter_vals:
                                selection.append({"attribute": att, "value": val})

                    syn_count = sdf.shape[0]
                    selection.sort(
                        key=lambda x: x["attribute"] + val_separator + x["value"]
                    )
                    
                    selection_key = att_separator.join(
                        [x["attribute"] + val_separator + x["value"] for x in selection]
                    )
                    filtered_aggs = adf[adf["selections"] == selection_key]

                    agg_records = adf[adf["selections"] == "record_count"][
                        "protected_count"
                    ].values[0]

                    if len(selection) == 0:
                        agg_count = agg_records
                    else:
                        agg_count = (
                            filtered_aggs["protected_count"].values[0]
                            if len(filtered_aggs) > 0
                            else None
                        )
                    best_est = agg_count if agg_count is not None else syn_count
                    # st.caption(count_intro)
                    perc = f"{best_est / agg_records:.1%}"
                    count_text = f"There are an estimated **{agg_records}** sensitive records overall."
                    if len(selection) > 0:
                        count_text = f"There are an estimated **{best_est}** sensitive records (**{perc}**) matching the query:\n\n{visuals.print_selections(selection)}"

                    count_holder.markdown(count_text)
                    st.markdown("##### Configure charts")
                    unit = st.text_input(
                        "Subject label",
                        value=st.session_state[f"{workflow}_unit"],
                        help='The type of data subject. For example, if the data is about people, the unit could be "Person".',
                    )
                    scheme = st.selectbox(
                        "Color scheme",
                        options=scheme_options,
                        index=scheme_options.index(
                            st.session_state[f"{workflow}_scheme"]
                        ),
                    )
                    s1, s2 = st.columns([1, 1])
                    with s1:
                        chart_width = st.number_input(
                            "Chart width",
                            value=st.session_state[f"{workflow}_chart_width"],
                        )
                    with s2:
                        chart_height = st.number_input(
                            "Chart height",
                            value=st.session_state[f"{workflow}_chart_height"],
                        )

                    chart = None
                    chart_df = None
                    chart_type = st.selectbox(
                        "Chart type",
                        options=chart_type_options,
                        index=chart_type_options.index(
                            st.session_state[f"{workflow}_chart_type"]
                            if f"{workflow}_chart_type" in st.session_state
                            else chart_type_options[0]
                        ),
                    )
                    if chart_type == "Top attributes":
                        if (
                            chart_type != st.session_state[f"{workflow}_chart_type"]
                            or st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]
                            == {}
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ] = {"show_attributes": [], "num_values": 10}
                            st.session_state[f"{workflow}_chart_type"] = chart_type
                            st.rerun()

                        chart_individual_configuration = st.session_state[
                            f"{workflow}_chart_individual_configuration"
                        ]
                        st.markdown("##### Configure top attributes chart")
                        print(
                            'chart_individual_configuration["show_attributes"]',
                            chart_individual_configuration["show_attributes"],
                        )
                        default_attrs = st.session_state[
                            f"{workflow}_chart_individual_configuration"
                        ]["show_attributes"]
                        # check if default attrs are in sdf.columns()
                        default_attrs_existing = [
                            attr
                            for attr in default_attrs
                            if attr in sdf.columns.to_numpy()
                        ]
                        show_attributes = st.multiselect(
                            "Types of top attributes to show",
                            options=sdf.columns.to_numpy(),
                            default=(
                                chart_individual_configuration["show_attributes"]
                                if (len(default_attrs_existing) == len(default_attrs))
                                else []
                            ),
                        )
                        num_values = st.number_input(
                            "Number of top attribute values to show",
                            value=chart_individual_configuration["num_values"],
                        )

                        if (
                            show_attributes
                            != st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["show_attributes"]
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["show_attributes"] = show_attributes
                            st.rerun()

                        if num_values != chart_individual_configuration["num_values"]:
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["num_values"] = num_values
                            st.rerun()

                        chart, chart_df = acd.get_bar_chart_fig(
                            selection=selection,
                            show_attributes=show_attributes,
                            unit=unit,
                            width=chart_width,
                            height=chart_height,
                            scheme=color_schemes[scheme],
                            num_values=num_values,
                            att_separator=config.att_separator,
                            val_separator=config.val_separator,
                        )
                    elif chart_type == "Time series":
                        if chart_type != st.session_state[f"{workflow}_chart_type"]:
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ] = {"time_attribute": "", "series_attributes": []}
                            st.session_state[f"{workflow}_chart_type"] = chart_type
                            st.rerun()

                        chart_individual_configuration = st.session_state[
                            f"{workflow}_chart_individual_configuration"
                        ]
                        st.markdown("##### Configure time series chart")
                        time_options = [""] + list(sdf.columns.values)
                        time_attribute = st.selectbox(
                            "Time attribute",
                            options=time_options,
                            index=time_options.index(chart_individual_configuration["time_attribute"])
                                if chart_individual_configuration["time_attribute"]
                                in sdf.columns.to_numpy()
                                else 0
                        )
                        series_attributes = st.multiselect(
                            "Series attributes",
                            options=list(sdf.columns.to_numpy())
                        )

                        if (
                            time_attribute
                            != chart_individual_configuration["time_attribute"]
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["time_attribute"] = time_attribute
                            st.rerun()

                        if (
                            time_attribute
                            != chart_individual_configuration["series_attributes"]
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["series_attributes"] = time_attribute
                            st.rerun()

                        if time_attribute != "" and len(series_attributes) > 0:
                            chart, chart_df = acd.get_line_chart_fig(
                                selection=selection,
                                series_attributes=series_attributes,
                                unit=unit,
                                time_attribute=time_attribute,
                                width=chart_width,
                                height=chart_height,
                                scheme=color_schemes[scheme],
                                att_separator=config.att_separator,
                                val_separator=config.val_separator,
                            )

                    elif chart_type == "Flow (alluvial)":
                        if chart_type != st.session_state[f"{workflow}_chart_type"]:
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ] = {
                                "source_attribute": "",
                                "target_attribute": "",
                                "highlight_attribute": "",
                            }
                            st.session_state[f"{workflow}_chart_type"] = chart_type
                            st.rerun()
                        chart_individual_configuration = st.session_state[
                            f"{workflow}_chart_individual_configuration"
                        ]
                        st.markdown("##### Configure flow (alluvial) chart")
                        attribute_type_options = [""] + list(sdf.columns.to_numpy())
                        highlight_options = ["", *options]
                        source_attribute_index = (
                            attribute_type_options.index(
                                chart_individual_configuration["source_attribute"]
                            )
                            if chart_individual_configuration["source_attribute"]
                            in attribute_type_options
                            else 0
                        )
                        source_attribute = st.selectbox(
                            "Source/origin attribute type",
                            options=attribute_type_options,
                            index=source_attribute_index,
                        )
                        target_attribute_index = (
                            attribute_type_options.index(
                                chart_individual_configuration["target_attribute"]
                            )
                            if chart_individual_configuration["target_attribute"]
                            in attribute_type_options
                            else 0
                        )
                        target_attribute = st.selectbox(
                            "Target/destination attribute type",
                            options=attribute_type_options,
                            index=target_attribute_index,
                        )
                        highlight_attribute_index = (
                            attribute_type_options.index(
                                chart_individual_configuration["highlight_attribute"]
                            )
                            if chart_individual_configuration["highlight_attribute"]
                            in attribute_type_options
                            else 0
                        )
                        highlight_attribute = st.selectbox(
                            "Highlight attribute",
                            options=highlight_options,
                            index=highlight_attribute_index,
                        )

                        if (
                            source_attribute
                            != chart_individual_configuration["source_attribute"]
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["source_attribute"] = source_attribute
                            st.rerun()

                        if (
                            target_attribute
                            != chart_individual_configuration["target_attribute"]
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["target_attribute"] = target_attribute
                            st.rerun()

                        if (
                            highlight_attribute
                            != chart_individual_configuration["highlight_attribute"]
                        ):
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ]["highlight_attribute"] = highlight_attribute
                            st.rerun()

                        if source_attribute != "" and target_attribute != "":
                            # export_df = compute_flow_query(selection, sv.anonymize_synthetic_df.value, adf, att_separator, val_separator, data_schema, source_attribute, target_attribute, highlight_attribute)
                            chart, chart_df = acd.get_flow_chart_fig(
                                selection=selection,
                                source_attribute=source_attribute,
                                target_attribute=target_attribute,
                                highlight_attribute=highlight_attribute,
                                unit=unit,
                                scheme=color_schemes[scheme],
                                width=chart_width,
                                height=chart_height,
                                att_separator=config.att_separator,
                                val_separator=config.val_separator
                            )

                    if chart_df is not None and chart is not None:
                        clear_btn = st.button("Clear configuration")
                        if clear_btn:
                            st.session_state[f"{workflow}_query_selections"] = []
                            st.session_state[f"{workflow}_unit"] = ""
                            st.session_state[f"{workflow}_scheme"] = scheme_options[0]
                            st.session_state[f"{workflow}_chart_width"] = 800
                            st.session_state[f"{workflow}_chart_height"] = 400
                            st.session_state[f"{workflow}_chart_type"] = (
                                chart_type_options[0]
                            )
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ] = {}
                            st.rerun()

                        st.markdown(
                            "##### Export",
                            help="Download the anonymized data and Plotly chart specification as CSV and JSON files, respectively.",
                        )
                        s1, s2 = st.columns([1, 1])
                        with s1:
                            st.download_button(
                                "Data CSV",
                                data=chart_df.to_csv(index=False),
                                file_name="data.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with s2:
                            st.download_button(
                                "Chart JSON",
                                data=pio.to_json(chart),
                                file_name="chart.json",
                                mime="text/json",
                                use_container_width=True,
                            )
                        # with s3:

                with container:
                    ad1, ad2 = st.columns([4, 1])
                    with ad1:
                        st.write(
                            "This page is not being cached. If you change workflows, you will need to re-configure your visualization."
                        )
                    with ad2:
                        cache = st.button("Cache visualization")
                        if cache:
                            st.session_state[f"{workflow}_query_selections"] = filters
                            st.session_state[f"{workflow}_unit"] = unit
                            st.session_state[f"{workflow}_scheme"] = scheme
                            st.session_state[f"{workflow}_chart_width"] = chart_width
                            st.session_state[f"{workflow}_chart_height"] = chart_height
                            st.session_state[f"{workflow}_chart_type"] = chart_type
                            st.session_state[
                                f"{workflow}_chart_individual_configuration"
                            ] = chart_individual_configuration
                            st.rerun()  #     st.download_button('Chart PNG', data=pio.to_image(chart, format='png'), file_name='chart.png', mime='image/png', use_container_width=True)

            with c2:
                st.markdown("##### Chart")
                if chart is not None:
                    st.plotly_chart(chart)

    with examples_tab:
        example_outputs_ui.create_example_outputs_ui(examples_tab, workflow)
       
