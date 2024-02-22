
import streamlit as st
import pandas as pd
import plotly.io as pio

import math
import os

from collections import defaultdict
from pacsynth import Dataset
from pacsynth import DpAggregateSeededParametersBuilder, AccuracyMode, FabricationMode
from pacsynth import DpAggregateSeededSynthesizer, Dataset

import workflows.data_synthesis.functions as functions
import workflows.data_synthesis.classes as classes
import workflows.data_synthesis.config as config
import workflows.data_synthesis.prompts as prompts
import workflows.data_synthesis.variables as vars

import util.ui_components
import util.df_functions


def create():
    workflow = 'data_synthesis'
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Data Synthesis')
    sv = vars.SessionVariables('data_synthesis')

    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    prepare_tab, generate_tab, queries_tab = st.tabs(['Prepare sensitive data', 'Generate synthetic data', 'Query and visualize'])
    df = None
    with prepare_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            util.ui_components.single_csv_uploader('Upload sensitive data CSV', sv.synthesis_last_sensitive_file_name, sv.synthesis_raw_sensitive_df, sv.synthesis_binned_df, key='sensitive_data_uploader', height=500)
        with model_col:
            util.ui_components.prepare_binned_df(workflow, sv.synthesis_raw_sensitive_df, sv.synthesis_binned_df, sv.synthesis_subject_identifier, sv.synthesis_min_count)
            sv.synthesis_process_columns.value = [col for col in sv.synthesis_raw_sensitive_df.value.columns.values if col != 'Subject ID' and st.session_state[f'{workflow}_{col}'] == True]
            
            for col in sv.synthesis_process_columns.value:
                print(sv.synthesis_binned_df.value[col].value_counts())
            
            distinct_counts = []
            num_cols = len(sv.synthesis_process_columns.value)
            
            for col in sv.synthesis_process_columns.value:
                distinct_values = tuple(sorted(sv.synthesis_binned_df.value[col].astype(str).unique()))
                if distinct_values == tuple(['0', '1']):
                    distinct_counts.append(1)
                else:
                    distinct_counts.append(len(distinct_values))
            # calculate number of pairs of column values using combinatorics
            num_observed_pairs = 0
            num_common_pairs = 0
            common_level = num_cols+1 + int(0.01 * pow(num_cols, 2)) #round(math.sqrt(num_cols))+1
            for ix, ci in enumerate(sv.synthesis_process_columns.value):
                for jx, cj in enumerate(sv.synthesis_process_columns.value[ix+1:]):
                    groups = sv.synthesis_binned_df.value[[ci, cj]].dropna().groupby([ci, cj]).size()

                    num_observed_pairs += len(groups)
                    # count groups with at least common_level records
                    common_groups = groups[groups >= common_level]
                    num_common_pairs += len(common_groups)
                    
            coverage = num_common_pairs / num_observed_pairs if num_observed_pairs > 0 else 1
            st.markdown(f'### Synthesizability summary')
            st.markdown(f'Number of selected columns: **{num_cols}**', help='This is the number of columns you selected for processing. The more columns you select, the harder it will be to synthesize data.')
            st.markdown(f'Common pair threshold: **{common_level}**', help='This is the minimum number of records that must appear in a pair of column values for the pair to be considered common. The higher this number, the harder it will be to synthesize data. The value is set as num_columns + 1 + int(0.01 * num_columns^2).')
            st.markdown(f'Estimated synthesizability score: **{round(coverage, 4)}**', help=f'We define synthesizability as the proportion of observed pairs of values across selected columns that are common, appearing at least as many times as the number of columns. In this case, {num_common_pairs}/{num_observed_pairs} pairs appear at least {num_cols} times. The intuition here is that all combinations of attribute values in a synthetic record must be composed from common attribute pairs. **Rule of thumb**: Aim for a synthesizability score of **0.5** or higher.')
    with generate_tab:
        if len(sv.synthesis_process_columns.value) == 0:
            st.markdown('Please upload and prepare data before generating synthetic data.')
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown(f'#### Synthesize data')
                b1, b2 = st.columns([1, 1])
                reporting_length = 4 # fixed
                with b1:
                    epsilon = st.number_input('Epsilon', value=sv.synthesis_epsilon.value, key=sv.synthesis_epsilon.key, help='The privacy budget, under differential privacy, to use when synthesizing the aggregate dataset.\n\nLower values of epsilon correspond to greater privacy protection but lower data quality.\n\nThe delta parameter is set automatically as 1/(protected_records*ln(protected_records)), where protected_records is the count of sensitive records protected using 0.5% of the privacy budget.\n\n**Rule of thumb**: Aim to keep epsilon at **12** or below.')
                with b2:
                    if st.button('Synthesize data'):
                        with st.spinner('Synthesizing data...'):
                            # Drop empty Subject ID rows
                            filtered = sv.synthesis_binned_df.value.dropna(subset=['Subject ID'])
                            filtered = sv.synthesis_binned_df.value[['Subject ID'] + sv.synthesis_process_columns.value]
                            melted = filtered.melt(id_vars=['Subject ID'], var_name='Attribute', value_name='Value').drop_duplicates()
                            att_to_subject_to_vals = defaultdict(lambda: defaultdict(set))
                            for i, row in melted.iterrows():
                                att_to_subject_to_vals[row['Attribute']][row['Subject ID']].add(row['Value'])
                            # define expanded atts as all attributes with more than one value for a given subject
                            expanded_atts = []
                            for att, subject_to_vals in att_to_subject_to_vals.items():
                                max_count = max([len(vals) for vals in subject_to_vals.values()])
                                if max_count > 1:
                                    expanded_atts.append(att)
                            new_rows = []
                            for i, row in melted.iterrows():
                                if row['Attribute'] in expanded_atts:
                                    if str(row['Value']) not in ['', '<NA>']:
                                        new_rows.append([row['Subject ID'], row['Attribute']+'_'+str(row['Value']), '1'])
                                else:
                                    new_rows.append([row['Subject ID'], row['Attribute'], str(row['Value'])])
                            melted = pd.DataFrame(new_rows, columns=['Subject ID', 'Attribute', 'Value'])
                            # convert back to wide format
                            sv.synthesis_sensitive_df.value = melted.pivot(index='Subject ID', columns='Attribute', values='Value').reset_index()
                            sv.synthesis_sensitive_df.value = sv.synthesis_sensitive_df.value.drop(columns=['Subject ID'])
                            sv.synthesis_sensitive_df.value.replace({'<NA>': ''}, inplace=True)

                            for col in sv.synthesis_sensitive_df.value.columns:
                                distinct_values = tuple(sorted(sv.synthesis_sensitive_df.value[col].astype(str).unique()))
                                if distinct_values == tuple(['0', '1']):
                                    sv.synthesis_sensitive_df.value.replace({col : {'0': ''}}, inplace=True)
                            sensitive_dataset = Dataset.from_data_frame(sv.synthesis_sensitive_df.value)
                            

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
                    st.markdown('**Caution**: These reports are intended only as a means to evaluate the quality of data for release. Releasing the values in these reports will compromise the privacy of the data. Do not release these reports or the values in them.')

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
            st.markdown('Please synthesize data before performing queries, or upload an existing synthetic dataset below.')
            util.ui_components.single_csv_uploader('Upload synthetic data CSV', sv.synthesis_last_synthetic_file_name, sv.synthesis_synthetic_df, sv.synthesis_synthetic_df, key='synthetic_data_uploader')
            util.ui_components.single_csv_uploader('Upload aggregate data CSV', sv.synthesis_last_aggregate_file_name, sv.synthesis_aggregate_df, sv.synthesis_aggregate_df, key='aggregate_data_uploader')
            if len(sv.synthesis_synthetic_df.value) > 0 and len(sv.synthesis_aggregate_df.value) > 0:
                st.rerun()
        else:
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
                    
                    filters = st.multiselect(label='Add attributes to query', options=options)
                    
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
                    unit = st.text_input('Subject label', value='', help='The type of data subject. For example, if the data is about people, the unit could be "Person".')
                    scheme = st.selectbox('Color scheme', options=sorted(config.color_schemes.keys()))
                    s1, s2 = st.columns([1, 1])
                    with s1:
                        chart_width = st.number_input('Chart width', value=800)
                    with s2:
                        chart_height = st.number_input('Chart height', value=400)
                    
                    chart = None
                    export_df = None
                    chart_type = st.selectbox('Chart type', options=['Top attributes', 'Time series', 'Flow (alluvial)'])
                    if chart_type == 'Top attributes':
                        st.markdown(f'##### Configure top attributes chart')     
                        show_attributes = st.multiselect('Types of top attributes to show', options=sdf.columns.values)
                        num_values = st.number_input('Number of top attribute values to show', value=10)


                        export_df = functions.compute_top_attributes_query(selection, sdf, adf, att_separator, val_separator, data_schema, show_attributes, num_values)
                        if len(export_df) > 0:
                            chart = functions.get_bar_chart(selection, show_attributes, unit, export_df, chart_width, chart_height, scheme)
                    elif chart_type == 'Time series':
                        st.markdown(f'##### Configure time series chart')    
                        time_attribute = st.selectbox('Time attribute', options=['']+list(sdf.columns.values))
                        series_attributes = st.multiselect('Series attributes', options=list(sdf.columns.values))
                        if time_attribute != '' and len(series_attributes) > 0:
                            export_df = functions.compute_time_series_query(selection, sv.synthesis_synthetic_df.value, adf, att_separator, val_separator, data_schema, time_attribute, series_attributes)
                            chart = functions.get_line_chart(selection, series_attributes, unit, export_df, time_attribute, chart_width, chart_height, scheme)
                    elif chart_type == 'Flow (alluvial)':    
                        st.markdown(f'##### Configure flow (alluvial) chart')    
                        source_attribute = st.selectbox('Source/origin attribute type', options=['']+list(sdf.columns.values))
                        target_attribute = st.selectbox('Target/destination attribute type', options=['']+list(sdf.columns.values))
                        highlight_attribute = st.selectbox('Highlight attribute', options=['']+options)

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
                        st.markdown(f'##### Export')
                        s1, s2 = st.columns([1, 1])
                        with s1:
                            st.download_button('Data CSV', data=export_df.to_csv(index=False), file_name='data.csv', mime='text/csv', use_container_width=True)
                        with s2:
                            st.download_button('Chart JSON', data=pio.to_json(chart), file_name='chart.json', mime='text/json', use_container_width=True)
                        # with s3:
                        #     st.download_button('Chart PNG', data=pio.to_image(chart, format='png'), file_name='chart.png', mime='image/png', use_container_width=True)

            with c2:
                st.markdown(f'##### Chart')
                if chart is not None:
                    st.plotly_chart(chart)

