# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
import streamlit as st
import pandas as pd
import plotly.io as pio

import math

from collections import defaultdict
from pacsynth import Dataset
from pacsynth import DpAggregateSeededParametersBuilder, AccuracyMode, FabricationMode
from pacsynth import DpAggregateSeededSynthesizer, Dataset

import workflows.data_synthesis.functions as functions
import workflows.data_synthesis.classes as classes
import workflows.data_synthesis.config as config
import workflows.data_synthesis.variables as vars

import util.ui_components
import util.df_functions

def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(file_path, 'r') as file:
        return file.read()

def create(sv: vars.SessionVariables, workflow: None):
    intro_tab, prepare_tab, generate_tab, queries_tab = st.tabs(['Data synthesis workflow:', 'Upload sensitive data', 'Generate anonymous data', 'Query and visualize data'])
    df = None
    with intro_tab:
        st.markdown(get_intro())
    with prepare_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            util.ui_components.single_csv_uploader(workflow, 'Upload sensitive data CSV', sv.synthesis_last_sensitive_file_name, sv.synthesis_raw_sensitive_df, sv.synthesis_processing_df, sv.synthesis_sensitive_df, uploader_key=sv.synthesis_upload_key.value, key='sensitive_data_uploader', height=400)
        with model_col:
            util.ui_components.prepare_input_df(workflow, sv.synthesis_raw_sensitive_df, sv.synthesis_processing_df, sv.synthesis_sensitive_df, sv.synthesis_subject_identifier)
            
            if len(sv.synthesis_sensitive_df.value) > 0:
                distinct_counts = []
                wdf = sv.synthesis_sensitive_df.value
                att_cols = [col for col in wdf.columns if col != 'Subject ID']
                num_cols = len(att_cols) 
                
                for col in wdf.columns.values:
                    if col == 'Subject ID':
                        continue
                    distinct_values = tuple(sorted(wdf[col].astype(str).unique()))
                    # if distinct_values == tuple(['0', '1']):
                    #     distinct_counts.append(1)
                    # else:
                    distinct_counts.append(len(distinct_values))
                distinct_counts.sort()
                common_level = max(distinct_counts[int(len(distinct_counts) * 0.5)], len(distinct_counts))
                overall_att_count = sum(distinct_counts)
                # calculate number of pairs of column values using combinatorics
                num_observed_pairs = 0
                num_common_pairs = 0
                for ix, ci in enumerate(att_cols):
                    for jx, cj in enumerate(att_cols[ix+1:]):
                        groups = wdf[[ci, cj]].dropna().groupby([ci, cj]).size()

                        num_observed_pairs += len(groups)
                        # count groups with at least common_level records
                        common_groups = groups[groups >= common_level]
                        num_common_pairs += len(common_groups)
                        
                coverage = num_common_pairs / num_observed_pairs if num_observed_pairs > 0 else 1
                st.markdown(f'### Synthesizability summary')
                st.markdown(f'Number of selected columns: **{num_cols}**', help='This is the number of columns you selected for processing. The more columns you select, the harder it will be to synthesize data.')
                st.markdown(f'Number of distinct attribute values: **{overall_att_count}**', help='This is the total number of distinct attribute values across all selected columns. The more distinct values, the harder it will be to synthesize data.')
                st.markdown(f'Common pair threshold: **{common_level}**', help='This is the minimum number of records that must appear in a pair of column values for the pair to be considered common. The higher this number, the harder it will be to synthesize data. The value is set as max(median value count, num selected columns).')
                st.markdown(f'Estimated synthesizability score: **{round(coverage, 4)}**', help=f'We define synthesizability as the proportion of observed pairs of values across selected columns that are common, appearing at least as many times as the number of columns. In this case, {num_common_pairs}/{num_observed_pairs} pairs appear at least {num_cols} times. The intuition here is that all combinations of attribute values in a synthetic record must be composed from common attribute pairs. **Rule of thumb**: Aim for a synthesizability score of **0.5** or higher.')
        
    with generate_tab:
        if len(sv.synthesis_sensitive_df.value) == 0:
            st.warning('Please upload and prepare data to continue.')
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown(f'#### Synthesize data')
                b1, b2 = st.columns([1, 1])
                reporting_length = 4 # fixed
                with b1:
                    epsilon = st.number_input('Epsilon', value=sv.synthesis_epsilon.value, help='The privacy budget, under differential privacy, to use when synthesizing the aggregate dataset.\n\nLower values of epsilon correspond to greater privacy protection but lower data quality.\n\nThe delta parameter is set automatically as 1/(protected_records*ln(protected_records)), where protected_records is the count of sensitive records protected using 0.5% of the privacy budget.\n\n**Rule of thumb**: Aim to keep epsilon at **12** or below.')
                with b2:
                    if st.button('Synthesize data'):
                        sv.synthesis_epsilon.value = epsilon
                        with st.spinner('Synthesizing data...'):
                            # for col in sv.synthesis_wide_sensitive_df.value.columns:
                            #     distinct_values = tuple(sorted(sv.synthesis_wide_sensitive_df.value[col].astype(str).unique()))
                            #     if distinct_values == tuple(['0', '1']):
                            #         sv.synthesis_sensitive_df.value.replace({col : {'0': ''}}, inplace=True)
                            df = sv.synthesis_sensitive_df.value.drop(columns=['Subject ID'])
                            df = util.df_functions.fix_null_ints(df).astype(str).replace('nan', '')
                            sensitive_dataset = Dataset.from_data_frame(df)
                            

                            params = DpAggregateSeededParametersBuilder() \
                                    .reporting_length(reporting_length) \
                                    .epsilon(epsilon) \
                                    .percentile_percentage(99) \
                                    .percentile_epsilon_proportion(0.01) \
                                    .accuracy_mode(AccuracyMode.prioritize_long_combinations()) \
                                    .number_of_records_epsilon_proportion(0.005) \
                                    .fabrication_mode(FabricationMode.balanced()) \
                                    .empty_value("") \
                                    .weight_selection_percentile(95) \
                                    .use_synthetic_counts(True) \
                                    .aggregate_counts_scale_factor(1.0) \
                                    .build()

                            synth = DpAggregateSeededSynthesizer(params)
                            
                            synth.fit(sensitive_dataset)
                            protected_number_of_records = synth.get_dp_number_of_records()
                            delta = 1.0 / (math.log(protected_number_of_records) * protected_number_of_records)
                            sv.synthesis_delta.value =  f'{delta:.2e}'
                            synthetic_raw_data = synth.sample()
                            synthetic_dataset = Dataset(synthetic_raw_data)
                            synthetic_df = Dataset.raw_data_to_data_frame(synthetic_raw_data)
                            sv.synthesis_synthetic_df.value = synthetic_df

                            sensitive_aggregates = sensitive_dataset.get_aggregates(reporting_length, ';')

                            # export the differentially private aggregates (internal to the synthesizer)
                            dp_aggregates = synth.get_dp_aggregates(';')

                            # generate aggregates from the synthetic data
                            synthetic_aggregates = synthetic_dataset.get_aggregates(reporting_length, ';')

                            sensitive_aggregates_parsed = {
                                tuple(agg.split(';')): count for (agg, count) in sensitive_aggregates.items()
                            }
                            dp_aggregates_parsed = {
                                tuple(agg.split(';')): count for (agg, count) in dp_aggregates.items()
                            }
                            synthetic_aggregates_parsed = {
                                tuple(agg.split(';')): count for (agg, count) in synthetic_aggregates.items()
                            }

                            agg_df = pd.DataFrame(data=dp_aggregates.items(), columns=['selections', 'protected_count'])
                            agg_df.loc[len(agg_df)] = ['record_count', protected_number_of_records]
                            agg_df = agg_df.sort_values(by=['protected_count'], ascending=False)
                            protected_number_of_records = synth.get_dp_number_of_records()
                            
                            sv.synthesis_aggregate_df.value = agg_df

                            sv.synthesis_sen_agg_rep.value = classes.ErrorReport(sensitive_aggregates_parsed, dp_aggregates_parsed).gen()
                            sv.synthesis_sen_syn_rep.value = classes.ErrorReport(sensitive_aggregates_parsed, synthetic_aggregates_parsed).gen()

                st.markdown(f'#### Analyze data', help='Tables show three evaluation metrics for each **Length** of attribute combination up to 4, plus an **Overall** average.\n\n- **Count +/- Error** is the average number of records for the combination length +/- the average absolute error in the number of records.\n- **Suppressed %** is the percentage of the total attribute counts that were suppressed, i.e., present in the Sensitive data but not the Aggregate/Synthetic data.\n- **Fabricated %** is the percentage of the total attribute counts that were fabricated, i.e., present in the Aggregate/Synthetic data but not the Sensitive data.\n\nPercentages are calculated with respect to attribute counts in the Sensitive data.\n\n**Rule of thumb**: For the Synthetic data, aim to keep the Overall Error below the Overall Count, Suppressed % below 10%, and Fabricated % below 1%.')
                
                if len(sv.synthesis_sen_agg_rep.value) > 0:
                    st.markdown(f'Differential privacy parameters: **Epsilon = {sv.synthesis_epsilon.value}**, **Delta = {sv.synthesis_delta.value}**')
                    st.markdown(f'###### Aggregate data quality')
                    st.dataframe(sv.synthesis_sen_agg_rep.value, hide_index=True, use_container_width=True)
                    st.markdown(f'###### Synthetic data quality')
                    st.dataframe(sv.synthesis_sen_syn_rep.value, hide_index=True, use_container_width=True)
                    st.warning('**Caution**: These tables should only be used to evaluate the quality of data for release. Sharing them compromises privacy.')

            with c2:
                st.markdown(f'##### Aggregate data')
                if len(sv.synthesis_aggregate_df.value) > 0:
                    st.dataframe(sv.synthesis_aggregate_df.value, hide_index=True, use_container_width=True, height=700)
                    st.download_button('Download Aggregate data', data=sv.synthesis_aggregate_df.value.to_csv(index=False), file_name='aggregate_data.csv', mime='text/csv')


            with c3:
                st.markdown(f'##### Synthetic data')
                if len(sv.synthesis_synthetic_df.value) > 0:
                    st.dataframe(sv.synthesis_synthetic_df.value, hide_index=True, use_container_width=True, height=700)
                    st.download_button('Download Synthetic data', data=sv.synthesis_synthetic_df.value.to_csv(index=False), file_name='synthetic_data.csv', mime='text/csv')
        
    with queries_tab:
        if len(sv.synthesis_synthetic_df.value) == 0 or len(sv.synthesis_aggregate_df.value) == 0:
            st.warning('Please synthesize data to continue, or upload an existing synthetic dataset below.')
            util.ui_components.single_csv_uploader(workflow, 'Upload synthetic data CSV', sv.synthesis_last_synthetic_file_name, sv.synthesis_synthetic_df, None, None, uploader_key=sv.synthesis_synthetic_upload_key.value,key='synthetic_data_uploader')
            util.ui_components.single_csv_uploader(workflow, 'Upload aggregate data CSV', sv.synthesis_last_aggregate_file_name, sv.synthesis_aggregate_df, None, None, uploader_key=sv.synthesis_aggregate_upload_key.value, key='aggregate_data_uploader')
            if len(sv.synthesis_synthetic_df.value) > 0 and len(sv.synthesis_aggregate_df.value) > 0:
                st.rerun()
        else:
            container = st.container(border=True)
            scheme_options = sorted(config.color_schemes.keys())
            chart_type_options = ['Top attributes', 'Time series', 'Flow (alluvial)']
            
            if f'{workflow}_query_selections' not in st.session_state:
                st.session_state[f'{workflow}_query_selections'] = []
            if f'{workflow}_unit' not in st.session_state:
                st.session_state[f'{workflow}_unit'] = ''
            if f'{workflow}_scheme' not in st.session_state:
                st.session_state[f'{workflow}_scheme'] = scheme_options[0]
            if f'{workflow}_chart_width' not in st.session_state:
                st.session_state[f'{workflow}_chart_width'] = 800
            if f'{workflow}_chart_height' not in st.session_state:
                st.session_state[f'{workflow}_chart_height'] = 400
            if f'{workflow}_chart_type' not in st.session_state:
                st.session_state[f'{workflow}_chart_type'] = chart_type_options[0]
            if f'{workflow}_chart_individual_configuration' not in st.session_state:
                st.session_state[f'{workflow}_chart_individual_configuration'] = {}
            if f'{workflow}_time_attributes' not in st.session_state:
                st.session_state[f'{workflow}_time_attributes'] = ''
            if f'{workflow}_series_attributes' not in st.session_state:
                st.session_state[f'{workflow}_series_attributes'] = []
                
            adf = sv.synthesis_aggregate_df.value
            adf['protected_count'] = adf['protected_count'].astype(int)
            sdf = sv.synthesis_synthetic_df.value.copy(deep=True)
            options = []
            for att in sdf.columns.values:
                vals = [f'{att}:{x}' for x in sdf[att].unique() if len(str(x)) > 0]
                vals.sort()
                options.extend(vals)
            c1, c2 = st.columns([1, 2])
            val_separator = ':'
            att_separator = ';'
            data_schema = defaultdict(list)
            data_schema_text = ''
            with c1:
                st.markdown(f'##### Constuct query')
                if len(sdf) > 0:
                    for att in sdf.columns.values:
                        vals = [str(x) for x in sdf[att].unique() if len(str(x)) > 0]
                        for val in vals:
                            data_schema[att].append(val)
                            data_schema_text += f'- {att} = {val}\n'
                        data_schema_text += '\n'
                        data_schema[att].sort()
                    count_holder = st.empty()
                    
                    filters = st.multiselect(label='Add attributes to query', options=options, default=st.session_state[f'{workflow}_query_selections'])

                    selection = []
                    for att, vals in data_schema.items():
                        filter_vals = [v for v in vals if f'{att}:{v}' in filters]
                        if len(filter_vals) > 0:
                            sdf = sdf[sdf[att].isin(filter_vals)]
                            for val in filter_vals:
                                selection.append({'attribute' : att, 'value' : val})

                    syn_count = sdf.shape[0]
                    selection.sort(key=lambda x: x['attribute']+val_separator+x['value'])
                    selection_key = att_separator.join([x['attribute']+val_separator+x['value'] for x in selection])
                    filtered_aggs = adf[adf['selections'] == selection_key]

                    agg_records = adf[adf['selections'] == 'record_count']['protected_count'].values[0]

                    if len(selection) == 0:
                        agg_count = agg_records
                    else:
                        agg_count = filtered_aggs['protected_count'].values[0] if len(filtered_aggs) > 0 else None
                    best_est = agg_count if agg_count is not None else syn_count
                    # st.caption(count_intro)
                    perc = f'{best_est / agg_records:.1%}'
                    count_text = f'There are an estimated **{agg_records}** sensitive records overall.'
                    if len(selection) > 0:
                        count_text = f'There are an estimated **{best_est}** sensitive records (**{perc}**) matching the query:\n\n{functions.print_selections(selection)}'

                    count_holder.markdown(count_text)
                    st.markdown(f'##### Configure charts')
                    unit = st.text_input('Subject label', value=st.session_state[f'{workflow}_unit'], help='The type of data subject. For example, if the data is about people, the unit could be "Person".')
                    scheme = st.selectbox('Color scheme', options=scheme_options, index=scheme_options.index(st.session_state[f'{workflow}_scheme']))
                    s1, s2 = st.columns([1, 1])
                    with s1:
                        chart_width = st.number_input('Chart width', value=st.session_state[f'{workflow}_chart_width'])
                    with s2:
                        chart_height = st.number_input('Chart height', value=st.session_state[f'{workflow}_chart_height'])
                    
                    chart = None
                    export_df = None
                    chart_type = st.selectbox('Chart type', options=chart_type_options, index=chart_type_options.index(st.session_state[f'{workflow}_chart_type']))
                    if chart_type == 'Top attributes':
                        if chart_type != st.session_state[f'{workflow}_chart_type'] or st.session_state[f'{workflow}_chart_individual_configuration'] == {}:
                            st.session_state[f'{workflow}_chart_individual_configuration'] = {
                                'show_attributes' : [],
                                'num_values' : 10
                            }
                            st.session_state[f'{workflow}_chart_type'] = chart_type
                            st.rerun()
                            
                        chart_individual_configuration = st.session_state[f'{workflow}_chart_individual_configuration']
                        st.markdown(f'##### Configure top attributes chart')     
                        show_attributes = st.multiselect('Types of top attributes to show', options=sdf.columns.values, default=chart_individual_configuration['show_attributes'])
                        num_values = st.number_input('Number of top attribute values to show', value=chart_individual_configuration['num_values'])
                        chart_individual_configuration['show_attributes'] = show_attributes
                        chart_individual_configuration['num_values'] = num_values
                        export_df = functions.compute_top_attributes_query(selection, sdf, adf, att_separator, val_separator, data_schema, show_attributes, num_values)
                        if len(export_df) > 0:
                            chart = functions.get_bar_chart(selection, show_attributes, unit, export_df, chart_width, chart_height, scheme)
                    elif chart_type == 'Time series':
                        if chart_type != st.session_state[f'{workflow}_chart_type']:
                            st.session_state[f'{workflow}_chart_individual_configuration'] = {
                                'time_attribute' : '',
                                'series_attributes' : []
                            }
                            st.session_state[f'{workflow}_chart_type'] = chart_type
                            st.rerun()

                        chart_individual_configuration = st.session_state[f'{workflow}_chart_individual_configuration']
                        st.markdown(f'##### Configure time series chart')    
                        time_options = ['']+list(sdf.columns.values)
                        time_attribute = st.selectbox('Time attribute', options=time_options, index=time_options.index(chart_individual_configuration['time_attribute']))
                        series_attributes = st.multiselect('Series attributes', options=list(sdf.columns.values), default=chart_individual_configuration['series_attributes'])
                        chart_individual_configuration['time_attribute'] = time_attribute
                        chart_individual_configuration['series_attributes'] = series_attributes
                        
                        if time_attribute != '' and len(series_attributes) > 0:
                            export_df = functions.compute_time_series_query(selection, sv.synthesis_synthetic_df.value, adf, att_separator, val_separator, data_schema, time_attribute, series_attributes)
                            chart = functions.get_line_chart(selection, series_attributes, unit, export_df, time_attribute, chart_width, chart_height, scheme)
                    elif chart_type == 'Flow (alluvial)':    
                        if chart_type != st.session_state[f'{workflow}_chart_type']:
                            st.session_state[f'{workflow}_chart_individual_configuration'] = {
                                'source_attribute' : '',
                                'target_attribute' : '',
                                'highlight_attribute' : ''
                            }
                            st.session_state[f'{workflow}_chart_type'] = chart_type
                            st.rerun()
                        chart_individual_configuration = st.session_state[f'{workflow}_chart_individual_configuration']
                        st.markdown(f'##### Configure flow (alluvial) chart')    
                        attribute_type_options = ['']+list(sdf.columns.values)
                        highlight_options = ['']+options
                        source_attribute = st.selectbox('Source/origin attribute type', options=attribute_type_options, index=attribute_type_options.index(chart_individual_configuration['source_attribute']))
                        target_attribute = st.selectbox('Target/destination attribute type', options=attribute_type_options, index=attribute_type_options.index(chart_individual_configuration['target_attribute']))
                        highlight_attribute = st.selectbox('Highlight attribute', options=highlight_options, index=highlight_options.index(chart_individual_configuration['highlight_attribute']))
                        chart_individual_configuration['source_attribute'] = source_attribute
                        chart_individual_configuration['target_attribute'] = target_attribute
                        chart_individual_configuration['highlight_attribute'] = highlight_attribute

                        if source_attribute != '' and target_attribute != '':
                            # export_df = compute_flow_query(selection, sv.synthesis_synthetic_df.value, adf, att_separator, val_separator, data_schema, source_attribute, target_attribute, highlight_attribute)
                            att_count = 2 if highlight_attribute == '' else 3
                            att_count += len(filters)
                            if att_count <= 4:
                                export_df = functions.compute_aggregate_graph(adf, filters, source_attribute, target_attribute, highlight_attribute)
                            else:
                                export_df = functions.compute_synthetic_graph(sdf, filters, source_attribute, target_attribute, highlight_attribute)
                            chart = functions.flow_chart(export_df, selection, source_attribute, target_attribute, highlight_attribute, chart_width, chart_height, unit, scheme)

                    if export_df is not None and chart is not None:
                        clear_btn = st.button('Clear configuration')
                        if (clear_btn):
                            st.session_state[f'{workflow}_query_selections'] = []
                            st.session_state[f'{workflow}_unit'] = ''
                            st.session_state[f'{workflow}_scheme'] = scheme_options[0]
                            st.session_state[f'{workflow}_chart_width'] = 800
                            st.session_state[f'{workflow}_chart_height'] = 400
                            st.session_state[f'{workflow}_chart_type'] = chart_type_options[0]
                            st.session_state[f'{workflow}_chart_individual_configuration'] = {}
                            st.rerun()

                        st.markdown(f'##### Export', help='Download the anonymized data and Plotly chart specification as CSV and JSON files, respectively.')
                        s1, s2 = st.columns([1, 1])
                        with s1:
                            st.download_button('Data CSV', data=export_df.to_csv(index=False), file_name='data.csv', mime='text/csv', use_container_width=True)
                        with s2:
                            st.download_button('Chart JSON', data=pio.to_json(chart), file_name='chart.json', mime='text/json', use_container_width=True)
                        # with s3:
            
                with container:
                    ad1, ad2 = st.columns([4, 1])
                    with ad1:
                        st.write('This page is not being cached. If you change workflows, you will need to re-configure your visualization.')
                    with ad2:
                        cache = st.button('Cache visualization')
                        if cache:
                            st.session_state[f'{workflow}_query_selections'] = filters
                            st.session_state[f'{workflow}_unit'] = unit
                            st.session_state[f'{workflow}_scheme'] = scheme
                            st.session_state[f'{workflow}_chart_width'] = chart_width
                            st.session_state[f'{workflow}_chart_height'] = chart_height
                            st.session_state[f'{workflow}_chart_type'] = chart_type
                            st.session_state[f'{workflow}_chart_individual_configuration'] = chart_individual_configuration
                            st.rerun()            #     st.download_button('Chart PNG', data=pio.to_image(chart, format='png'), file_name='chart.png', mime='image/png', use_container_width=True)

            with c2:
                st.markdown(f'##### Chart')
                if chart is not None:
                    st.plotly_chart(chart)

