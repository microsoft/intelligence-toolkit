# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import math
import os
from collections import defaultdict

import pandas as pd
import plotly.io as pio
import streamlit as st
from pacsynth import (
    AccuracyMode,
    Dataset,
    DpAggregateSeededParametersBuilder,
    DpAggregateSeededSynthesizer,
    FabricationMode,
)

import app.util.df_functions as df_functions
import app.util.ui_components as ui_components
import app.workflows.anonymize_case_data.classes as classes
import app.workflows.anonymize_case_data.config as config
import app.workflows.anonymize_case_data.functions as functions
import app.workflows.anonymize_case_data.variables as ds_variables
from app.tutorials import get_tutorial
from toolkit.anonymize_code_data import get_readme as get_intro


def create(sv: ds_variables.SessionVariables, workflow: None):
    intro_tab, prepare_tab, generate_tab, queries_tab, examples_tab = st.tabs(
        [
            "Anonymize case data workflow:",
            "Prepare sensitive data",
            "Generate anonymous data",
            "Query and visualize data",
            "View example outputs"
        ]
    )
    df = None
    with intro_tab:
        st.markdown(get_intro() + get_tutorial(workflow))
    with prepare_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            ui_components.single_csv_uploader(
                workflow,
                "Upload sensitive data CSV",
                sv.anonymize_last_sensitive_file_name,
                sv.anonymize_raw_sensitive_df,
                sv.anonymize_processing_df,
                sv.anonymize_sensitive_df,
                uploader_key=sv.anonymize_upload_key.value,
                key="sensitive_data_uploader",
                height=400,
            )
        with model_col:
            ui_components.prepare_input_df(
                workflow,
                sv.anonymize_raw_sensitive_df,
                sv.anonymize_processing_df,
                sv.anonymize_sensitive_df,
                sv.anonymize_subject_identifier,
            )

            if len(sv.anonymize_sensitive_df.value) > 0:
                distinct_counts = []
                wdf = sv.anonymize_sensitive_df.value
                att_cols = [col for col in wdf.columns if col != "Subject ID"]
                num_cols = len(att_cols)
                print(att_cols)
                for col in wdf.columns.to_numpy():
                    if col == "Subject ID":
                        continue
                    distinct_values = tuple(sorted(wdf[col].astype(str).unique()))
                    # if distinct_values == tuple(['0', '1']):
                    #     distinct_counts.append(1)
                    # else:
                    distinct_counts.append(len(distinct_values))
                print(distinct_counts)
                distinct_counts.sort()
                overall_att_count = sum(distinct_counts)
                possible_combinations = math.prod(distinct_counts)
                possible_combinations_per_row = int(round(possible_combinations / wdf.shape[0]))
                max_combinations_per_record = 2**len(att_cols)
                excess_combinations_ratio = possible_combinations_per_row / max_combinations_per_record
                st.markdown("### Synthesizability summary")
                st.markdown(
                    f"Number of selected columns: **{num_cols}**",
                    help="This is the number of columns you selected for processing. The more columns you select, the harder it will be to synthesize data.",
                )
                st.markdown(
                    f"Number of distinct attribute values: **{overall_att_count}**",
                    help="This is the total number of distinct attribute values across all selected columns. The more distinct values, the harder it will be to synthesize data.",
                )
                st.markdown(
                    f"Number of possible combinations: **{possible_combinations}**",
                    help="This is the total number of possible attribute combinations across all selected columns. The higher this number, the harder it will be to synthesize data.",
                )
                st.markdown(
                    f"Mean combinations per record: **{possible_combinations_per_row}**",
                    help="This is the mean number of possible attribute combinations per sensitive case record. The higher this number, the harder it will be to synthesize data.",
                )
                st.markdown(
                    f"Maximum combinations per record: **{max_combinations_per_record}**",
                    help="This is the maximum number of possible attribute combinations per record. The higher this number, the harder it will be to synthesize data.",
                )
                st.markdown(
                    f"Excess combinations ratio: **{round(excess_combinations_ratio, 1)}**",
                    help="This is the ratio of possible combinations per record to the maximum possible combinations per record. The higher this number, the harder it will be to synthesize data. **Rule of thumb**: Aim for a ratio of **5** or lower.",
                )

    with generate_tab:
        if len(sv.anonymize_sensitive_df.value) == 0:
            st.warning("Please upload and prepare data to continue.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown("#### Anonymize data")
                b1, b2 = st.columns([1, 1])
                reporting_length = 4  # fixed
                with b1:
                    epsilon = st.number_input(
                        "Epsilon",
                        value=sv.anonymize_epsilon.value,
                        help="The privacy budget, under differential privacy, to use when synthesizing the aggregate dataset.\n\nLower values of epsilon correspond to greater privacy protection but lower data quality.\n\nThe delta parameter is set automatically as 1/(protected_records*ln(protected_records)), where protected_records is the count of sensitive records protected using 0.5% of the privacy budget.\n\n**Rule of thumb**: Aim to keep epsilon at **12** or below.",
                    )
                with b2:
                    if st.button("Anonymize data"):
                        sv.anonymize_epsilon.value = epsilon
                        with st.spinner("Anonymizing data..."):
                            # for col in sv.anonymize_wide_sensitive_df.value.columns:
                            #     distinct_values = tuple(sorted(sv.anonymize_wide_sensitive_df.value[col].astype(str).unique()))
                            #     if distinct_values == tuple(['0', '1']):
                            #         sv.anonymize_sensitive_df.value.replace({col : {'0': ''}}, inplace=True)
                            df = sv.anonymize_sensitive_df.value.drop(
                                columns=["Subject ID"]
                            )
                            df = (
                                df_functions.fix_null_ints(df)
                                .astype(str)
                                .replace("nan", "")
                            )
                            sensitive_dataset = Dataset.from_data_frame(df)

                            params = (
                                DpAggregateSeededParametersBuilder()
                                .reporting_length(reporting_length)
                                .epsilon(epsilon)
                                .percentile_percentage(99)
                                .percentile_epsilon_proportion(0.01)
                                .accuracy_mode(
                                    AccuracyMode.prioritize_long_combinations()
                                )
                                .number_of_records_epsilon_proportion(0.005)
                                .fabrication_mode(FabricationMode.balanced())
                                .empty_value("")
                                .weight_selection_percentile(95)
                                .use_synthetic_counts(True)
                                .aggregate_counts_scale_factor(1.0)
                                .build()
                            )

                            synth = DpAggregateSeededSynthesizer(params)

                            synth.fit(sensitive_dataset)
                            protected_number_of_records = (
                                synth.get_dp_number_of_records()
                            )
                            delta = 1.0 / (
                                math.log(protected_number_of_records)
                                * protected_number_of_records
                            )
                            sv.anonymize_delta.value = f"{delta:.2e}"
                            synthetic_raw_data = synth.sample()
                            synthetic_dataset = Dataset(synthetic_raw_data)
                            synthetic_df = Dataset.raw_data_to_data_frame(
                                synthetic_raw_data
                            )
                            sv.anonymize_synthetic_df.value = synthetic_df

                            sensitive_aggregates = sensitive_dataset.get_aggregates(
                                reporting_length, ";"
                            )

                            # export the differentially private aggregates (internal to the synthesizer)
                            dp_aggregates = synth.get_dp_aggregates(";")

                            # generate aggregates from the synthetic data
                            synthetic_aggregates = synthetic_dataset.get_aggregates(
                                reporting_length, ";"
                            )

                            sensitive_aggregates_parsed = {
                                tuple(agg.split(";")): count
                                for (agg, count) in sensitive_aggregates.items()
                            }
                            dp_aggregates_parsed = {
                                tuple(agg.split(";")): count
                                for (agg, count) in dp_aggregates.items()
                            }
                            synthetic_aggregates_parsed = {
                                tuple(agg.split(";")): count
                                for (agg, count) in synthetic_aggregates.items()
                            }

                            agg_df = pd.DataFrame(
                                data=dp_aggregates.items(),
                                columns=["selections", "protected_count"],
                            )
                            agg_df.loc[len(agg_df)] = [
                                "record_count",
                                protected_number_of_records,
                            ]
                            agg_df = agg_df.sort_values(
                                by=["protected_count"], ascending=False
                            )
                            protected_number_of_records = (
                                synth.get_dp_number_of_records()
                            )

                            sv.anonymize_aggregate_df.value = agg_df

                            sv.anonymize_sen_agg_rep.value = classes.ErrorReport(
                                sensitive_aggregates_parsed, dp_aggregates_parsed
                            ).gen()
                            sv.anonymize_sen_syn_rep.value = classes.ErrorReport(
                                sensitive_aggregates_parsed, synthetic_aggregates_parsed
                            ).gen()

                st.markdown(
                    "#### Analyze data",
                    help="Tables show three evaluation metrics for each **Length** of attribute combination up to 4, plus an **Overall** average.\n\n- **Count +/- Error** is the average number of records for the combination length +/- the average absolute error in the number of records.\n- **Suppressed %** is the percentage of the total attribute counts that were suppressed, i.e., present in the Sensitive data but not the Aggregate/Synthetic data.\n- **Fabricated %** is the percentage of the total attribute counts that were fabricated, i.e., present in the Aggregate/Synthetic data but not the Sensitive data.\n\nPercentages are calculated with respect to attribute counts in the Sensitive data.\n\n**Rule of thumb**: For the Synthetic data, aim to keep the Overall Error below the Overall Count, Suppressed % below 10%, and Fabricated % below 1%.",
                )

                if len(sv.anonymize_sen_agg_rep.value) > 0:
                    st.markdown(
                        f"Differential privacy parameters: **Epsilon = {sv.anonymize_epsilon.value}**, **Delta = {sv.anonymize_delta.value}**"
                    )
                    st.markdown("###### Aggregate data quality")
                    st.dataframe(
                        sv.anonymize_sen_agg_rep.value,
                        hide_index=True,
                        use_container_width=False,
                    )
                    st.markdown("###### Synthetic data quality")
                    st.dataframe(
                        sv.anonymize_sen_syn_rep.value,
                        hide_index=True,
                        use_container_width=False,
                    )
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
                None,
                uploader_key=sv.anonymize_synthetic_upload_key.value,
                key="synthetic_data_uploader",
            )
            ui_components.single_csv_uploader(
                workflow,
                "Upload aggregate data CSV",
                sv.anonymize_last_aggregate_file_name,
                sv.anonymize_aggregate_df,
                None,
                None,
                uploader_key=sv.anonymize_aggregate_upload_key.value,
                key="aggregate_data_uploader",
            )
            if (
                len(sv.anonymize_synthetic_df.value) > 0
                and len(sv.anonymize_aggregate_df.value) > 0
            ):
                st.rerun()
        else:
            container = st.container(border=True)
            scheme_options = sorted(config.color_schemes.keys())
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
            data_schema = defaultdict(list)
            data_schema_text = ""
            with c1:
                st.markdown("##### Constuct query")
                if len(sdf) > 0:
                    for att in sdf.columns.to_numpy():
                        vals = [str(x) for x in sdf[att].unique() if len(str(x)) > 0]
                        for val in vals:
                            data_schema[att].append(val)
                            data_schema_text += f"- {att} = {val}\n"
                        data_schema_text += "\n"
                        data_schema[att].sort()
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
                        count_text = f"There are an estimated **{best_est}** sensitive records (**{perc}**) matching the query:\n\n{functions.print_selections(selection)}"

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
                    export_df = None
                    chart_type = st.selectbox(
                        "Chart type",
                        options=chart_type_options,
                        index=chart_type_options.index(
                            st.session_state[f"{workflow}_chart_type"]
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
                        show_attributes = st.multiselect(
                            "Types of top attributes to show",
                            options=sdf.columns.to_numpy(),
                            default=chart_individual_configuration["show_attributes"],
                        )
                        num_values = st.number_input(
                            "Number of top attribute values to show",
                            value=chart_individual_configuration["num_values"],
                        )
                        chart_individual_configuration["show_attributes"] = (
                            show_attributes
                        )
                        chart_individual_configuration["num_values"] = num_values
                        export_df = functions.compute_top_attributes_query(
                            selection,
                            sdf,
                            adf,
                            att_separator,
                            val_separator,
                            data_schema,
                            show_attributes,
                            num_values,
                        )
                        if len(export_df) > 0:
                            chart = functions.get_bar_chart(
                                selection,
                                show_attributes,
                                unit,
                                export_df,
                                chart_width,
                                chart_height,
                                scheme,
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
                            index=time_options.index(
                                chart_individual_configuration["time_attribute"]
                            ),
                        )
                        series_attributes = st.multiselect(
                            "Series attributes",
                            options=list(sdf.columns.to_numpy())
                        )
                        chart_individual_configuration["time_attribute"] = (
                            time_attribute
                        )
                        chart_individual_configuration["series_attributes"] = (
                            series_attributes
                        )

                        if time_attribute != "" and len(series_attributes) > 0:
                            export_df = functions.compute_time_series_query(
                                selection,
                                sv.anonymize_synthetic_df.value,
                                adf,
                                att_separator,
                                val_separator,
                                data_schema,
                                time_attribute,
                                series_attributes,
                            )
                            chart = functions.get_line_chart(
                                selection,
                                series_attributes,
                                unit,
                                export_df,
                                time_attribute,
                                chart_width,
                                chart_height,
                                scheme,
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
                        source_attribute = st.selectbox(
                            "Source/origin attribute type",
                            options=attribute_type_options,
                            index=attribute_type_options.index(
                                chart_individual_configuration["source_attribute"]
                            ),
                        )
                        target_attribute = st.selectbox(
                            "Target/destination attribute type",
                            options=attribute_type_options,
                            index=attribute_type_options.index(
                                chart_individual_configuration["target_attribute"]
                            ),
                        )
                        highlight_attribute = st.selectbox(
                            "Highlight attribute",
                            options=highlight_options,
                            index=highlight_options.index(
                                chart_individual_configuration["highlight_attribute"]
                            ),
                        )
                        chart_individual_configuration["source_attribute"] = (
                            source_attribute
                        )
                        chart_individual_configuration["target_attribute"] = (
                            target_attribute
                        )
                        chart_individual_configuration["highlight_attribute"] = (
                            highlight_attribute
                        )

                        if source_attribute != "" and target_attribute != "":
                            # export_df = compute_flow_query(selection, sv.anonymize_synthetic_df.value, adf, att_separator, val_separator, data_schema, source_attribute, target_attribute, highlight_attribute)
                            att_count = 2 if highlight_attribute == "" else 3
                            att_count += len(filters)
                            if att_count <= 4:
                                export_df = functions.compute_aggregate_graph(
                                    adf,
                                    filters,
                                    source_attribute,
                                    target_attribute,
                                    highlight_attribute,
                                )
                            else:
                                export_df = functions.compute_anonym_graph(
                                    sdf,
                                    filters,
                                    source_attribute,
                                    target_attribute,
                                    highlight_attribute,
                                )
                            chart = functions.flow_chart(
                                export_df,
                                selection,
                                source_attribute,
                                target_attribute,
                                highlight_attribute,
                                chart_width,
                                chart_height,
                                unit,
                                scheme,
                            )

                    if export_df is not None and chart is not None:
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
                                data=export_df.to_csv(index=False),
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
