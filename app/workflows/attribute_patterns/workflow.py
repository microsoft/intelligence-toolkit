import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from st_aggrid import (
    AgGrid,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)

import os

import workflows.attribute_patterns.functions as functions
import workflows.attribute_patterns.classes as classes
import workflows.attribute_patterns.config as config
import workflows.attribute_patterns.prompts as prompts
import workflows.attribute_patterns.variables as vars

def create():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Attribute Patterns')
    sv = vars.SessionVariables('attribute_patterns')

    uploader_tab, process_tab, patterns_tab = st.tabs(['Create graph model', 'Embed graph model', 'Detect patterns'])
    df = None
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            st.markdown('##### Upload data for processing')
            files = st.file_uploader("Upload CSVs", type=['csv'], accept_multiple_files=True)
            st.number_input('Maximum rows to process (0 = all)', min_value=0, max_value=1000000, step=1000, key=sv.attribute_max_rows_to_process.key)
            
            if files != None:
                for file in files:
                    if file.name not in sv.attribute_uploaded_files.value:
                        df = pd.read_csv(file, encoding='unicode_escape')[:sv.attribute_max_rows_to_process.value] if sv.attribute_max_rows_to_process.value > 0 else pd.read_csv(file, encoding='unicode_escape')
                        df.to_csv(os.path.join(config.outputs_dir, file.name), index=False)
                        sv.attribute_uploaded_files.value.append(file.name)
            selected_file = st.selectbox("Select a file", sv.attribute_uploaded_files.value)
            
            if selected_file != None:
                df = pd.read_csv(selected_file, encoding='unicode_escape')[:sv.attribute_max_rows_to_process.value] if sv.attribute_max_rows_to_process.value > 0 else pd.read_csv(selected_file, encoding='unicode_escape')
                df = df.fillna('').astype(str)
                st.dataframe(df[:config.max_rows_to_show], hide_index=True, use_container_width=True)
        with model_col:
                st.markdown('##### Map columns to graph model')
                if df is None:
                    st.markdown('Upload and select a file to continue')
                else:
                    # Map all data into a single model of Period | Entity ID | Attribute Type | Attribute Value | Full Attribute
                    att_types = []
                    cols = [''] + df.columns.values.tolist()
                    data_format = st.radio('Data format', ['Long (1 line/attribute)', 'Wide (1 line/record)'], horizontal=True)
                    entity_col = ''
                    type_col = ''
                    val_col = ''
                    time_col = ''
                    ready = False
                    if data_format == 'Long (1 line/attribute)':
                        entity_col = st.selectbox('Entity ID column', cols)
                        att_form = st.radio('Attribute format', ['Separate type and value columns', 'Combined type-value column'], horizontal=True)
                        if att_form == 'Separate type and value columns':
                            type_col = st.selectbox('Attribute type column', cols)
                            if type_col != '':
                                att_types = sorted(df[type_col].unique().tolist())
                            val_col = st.selectbox('Attribute value column', cols)
                        else:
                            type_val_col = st.selectbox('Combined type-value column', cols)
                            type_val_sep_in = st.text_input('Type-value separator', key=sv.attribute_type_val_sep_in.key, value=sv.attribute_type_val_sep_in.value)
                            att_types = sorted(df[type_val_col].apply(lambda x: x.split(type_val_sep_in)[0]).unique().tolist()) if type_val_col != '' and type_val_sep_in != '' else []
                        time_format = st.radio('Time period format', ['Period column', 'Period attribute type'], horizontal=True)
                        if time_format == 'Period column':
                            time_col = st.selectbox('Period column', cols)
                        else:
                            time_att = st.selectbox('Period attribute type', [''] + att_types)
                        ready = entity_col != '' and ((att_form == 'Separate type and value columns' and type_col != '' and val_col != '') or (att_form == 'Combined type-value column' and type_val_col != '' and type_val_sep_in != '')) and ((time_format == 'Period column' and time_col != '') or (time_format == 'Period attribute type' and time_att != ''))
                    elif data_format == 'Wide (1 line/record)':
                        att_cols = st.multiselect('Attribute columns', cols)
                        time_col = st.selectbox('Period column', cols)
                        ready = len(att_cols) > 0 and time_col != ''

                    if st.button("Add links to graph model", disabled=not ready):
                        with st.spinner('Adding links to model...'):
                            if data_format == 'Long (1 line/attribute)':
                                pdf = df.copy(deep=True)
                                pdf.rename(columns={entity_col : 'Entity ID'}, inplace=True)
                                if att_form == 'Separate type and value columns':
                                    pdf.rename(columns={type_col : 'Attribute Type', val_col : 'Attribute Value'}, inplace=True)
                                    pdf['Full Attribute'] = pdf['Attribute Type'] + sv.attribute_type_val_sep_out.value + pdf['Attribute Value']
                                elif att_form == 'Combined type-value column':
                                    pdf['Attribute Type'] = pdf[type_val_col].apply(lambda x: x.split(type_val_sep_in)[0])
                                    pdf['Attribute Value'] = pdf[type_val_col].apply(lambda x: x.split(type_val_sep_in)[1])
                                    pdf['Full Attribute'] = pdf['Attribute Type'] + type_val_sep_in + pdf['Attribute Value']
                                if time_format == 'Period column':
                                    pdf['Period'] = pdf[time_col]
                                else:
                                    entity_times = pdf[pdf['Attribute Type'] == time_att] [['Entity ID', 'Attribute Value']].drop_duplicates()   
                                    entity_times.rename(columns={'Attribute Value' : 'Period'}, inplace=True)                         
                                    # join on entity ID
                                    pdf = pdf.merge(entity_times, on='Entity ID', how='inner')
                                    pdf = pdf[pdf['Attribute Type'] != time_att]
                                pdf = pdf[['Period', 'Entity ID', 'Attribute Type', 'Attribute Value', 'Full Attribute']]
                            elif data_format == 'Wide (1 line/record)':
                                if entity_col == '':
                                    entity_col = 'Entity ID'
                                    df[entity_col] = df.index.astype(str)
                                pdf = df.copy(deep=True)[[time_col, entity_col] + att_cols]
                                pdf = pdf[pdf[time_col].notna() & pdf[entity_col].notna()]
                                pdf.rename(columns={entity_col : 'Entity ID', time_col : 'Period'}, inplace=True)
                                pdf = pd.melt(pdf, id_vars=['Entity ID', 'Period'], value_vars=att_cols, var_name='Attribute Type', value_name='Attribute Value')
                                pdf = pdf[pdf['Attribute Value'].notna()]
                                pdf['Full Attribute'] = pdf.apply(lambda x: str(x['Attribute Type']) + sv.attribute_type_val_sep_out.value + str(x['Attribute Value']), axis=1)
                        sv.attribute_dynamic_df.value = pdf
                    if len(sv.attribute_dynamic_df.value) > 0:
                        st.markdown(f'Graph model has **{len(sv.attribute_dynamic_df.value)}** links spanning **{len(sv.attribute_dynamic_df.value["Entity ID"].unique())}** cases, **{len(sv.attribute_dynamic_df.value["Full Attribute"].unique())}** attributes, and **{len(sv.attribute_dynamic_df.value["Period"].unique())}** periods.')
    with process_tab:
        if sv.attribute_dynamic_df.value is None:
            st.markdown('Upload and select a file to continue')
        else:
            c1, c2 = st.columns([1, 4])
            with c1:
                st.markdown('##### Configure GEE model')
                st.number_input('Minimum edge weight', min_value=0.0, max_value=1.0, step=0.001, key=sv.attribute_min_edge_weight.key, value=sv.attribute_min_edge_weight.value, format='%.4f')
                st.number_input('Missing weight as proportion of minimum weight', min_value=0.0, max_value=1.0, step=0.01, key=sv.attribute_missing_edge_prop.key, value=sv.attribute_missing_edge_prop.value, format='%.2f')
                st.checkbox('Use Laplacian', key=sv.attribute_laplacian.key, value=sv.attribute_laplacian.value)
                st.checkbox('Use Diagonal A', key=sv.attribute_diaga.key, value=sv.attribute_diaga.value)
                st.checkbox('Use Correlation', key=sv.attribute_correlation.key, value=sv.attribute_correlation.value)
                st.radio('Edge definition', ['Count', 'Mutual information', ], key='attribute_edge_definition', index=0)
                st.button('Detect structure', key='detect_attribute_network_structure', use_container_width=True)
            with c2:
                if st.session_state['detect_attribute_network_structure']:
                    with st.spinner('Detecting structure...'):
                        sv.attribute_df.value, time_to_graph = functions.prepare_graph(sv, mi=st.session_state['attribute_edge_definition'] == 'Mutual information')
                        sv.attribute_embedding_df.value, sv.attribute_node_to_centroid.value, sv.attribute_period_embeddings.value = functions.generate_embedding(sv, sv.attribute_df.value, time_to_graph)
                        sv.attribute_umap_df.value = functions.generate_umap(sv, sv.attribute_embedding_df.value)
                            
                umap_df = sv.attribute_umap_df.value
                if len(umap_df) > 0:
                    st.markdown('##### GEE Model (Color = Attribute Type)')
                    sp = functions.get_scatterplot(umap_df, 800)
                    st.altair_chart(sp, use_container_width=True)

    with patterns_tab:
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown('###### Primary patterns')
            st.markdown('Detected by clustering nearby attributes.')
            st.number_input('Minimum primary pattern count', min_value=1, max_value=1000000, step=1, key=sv.attribute_min_primary_pattern_count.key, value=sv.attribute_min_primary_pattern_count.value)
            if st.button('Detect primary patterns'):
                with st.spinner('Detecting primary patterns...'):
                    rc = classes.RecordCounter(sv.attribute_dynamic_df.value)
                    sv.attribute_record_counter.value = rc
                    sv.attribute_primary_pattern_df.value = functions.detect_primary_patterns(sv)
                    sv.attribute_primary_pattern_df.value['pattern_type'] = 'primary'
            st.markdown('###### Secondary patterns')
            st.markdown('Detected by combining converging attributes.')
            st.number_input('Minimum secondary pattern count', min_value=1, step=1, key=sv.attribute_min_secondary_pattern_count.key, value=sv.attribute_min_secondary_pattern_count.value)
            st.number_input('Maximum secondary pattern length', min_value=1, step=1, key=sv.attribute_max_secondary_pattern_length.key, value=sv.attribute_max_secondary_pattern_length.value)
            if st.button('Detect secondary patterns'):
                with st.spinner('Detecting secondary patterns...'):
                    rc = classes.RecordCounter(sv.attribute_dynamic_df.value)
                    sv.attribute_record_counter.value = rc
                    sv.attribute_secondary_pattern_df.value, sv.attribute_close_pairs.value, sv.attribute_all_pairs.value = functions.detect_secondary_patterns(sv)
                    sv.attribute_secondary_pattern_df.value['pattern_type'] = 'secondary'
                    
                
        with c2:
            sv.attribute_overall_pattern_df.value = pd.concat([sv.attribute_primary_pattern_df.value, sv.attribute_secondary_pattern_df.value])
                    
            if len(sv.attribute_overall_pattern_df.value) > 0:
                prop = sv.attribute_close_pairs.value / sv.attribute_all_pairs.value if sv.attribute_all_pairs.value > 0 else 0
                prop = round(prop, -int(np.floor(np.log10(abs(prop))))) if prop > 0 else 0
                period_count = len(sv.attribute_overall_pattern_df.value["period"].unique())
                primary_count = len(sv.attribute_overall_pattern_df.value[sv.attribute_overall_pattern_df.value['pattern_type'] == 'primary'])
                secondary_count = len(sv.attribute_overall_pattern_df.value[sv.attribute_overall_pattern_df.value['pattern_type'] == 'secondary'])
                st.markdown(f'Over **{period_count}** periods, detected **{primary_count}** primary attribute patterns and **{secondary_count}** secondary attribute patterns.')
                show_df = sv.attribute_overall_pattern_df.value
                tdf = functions.create_time_series_df(sv.attribute_record_counter.value, sv.attribute_overall_pattern_df.value)
                gb = GridOptionsBuilder.from_dataframe(show_df)
                gb.configure_default_column(wrapText=True, enablePivot=False, enableValue=False, enableRowGroup=False)
                gb.configure_selection(selection_mode="single", use_checkbox=False)
                gb.configure_side_bar()
                gridoptions = gb.build()

                response = AgGrid(
                    show_df,
                    key='report_grid',
                    height=250,
                    gridOptions=gridoptions,
                    enable_enterprise_modules=False,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=True,
                    header_checkbox_selection_filtered_only=False,
                    use_checkbox=False,
                    enable_quicksearch=True,
                    reload_data=True
                    ) # type: ignore
                selected_pattern = response['selected_rows'][0]['pattern'] if len(response['selected_rows']) > 0 else ''
                selected_pattern_period = response['selected_rows'][0]['period'] if len(response['selected_rows']) > 0 else ''

                if selected_pattern != '':

                    st.markdown('**Selected pattern: ' + selected_pattern + ' (' + selected_pattern_period + ')**')
                    tdf = tdf[tdf['pattern'] == selected_pattern]

                    count_ct = alt.Chart(tdf).mark_line().encode(
                        x='period:O',
                        y='count:Q',
                        color=alt.ColorValue('blue')
                    ).properties(
                        height = 200,
                        width = 600
                    )
                    st.altair_chart(count_ct, use_container_width=True)
