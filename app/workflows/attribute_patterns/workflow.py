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

import util.ui_components

def create():
    workflow = 'attribute_patterns'
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Attribute Patterns')
    sv = vars.SessionVariables('attribute_patterns')

    uploader_tab, process_tab, patterns_tab = st.tabs(['Create graph model', 'Embed graph model', 'Detect patterns'])
    df = None
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            util.ui_components.single_csv_uploader('Upload CSV', sv.attribute_last_file_name, sv.attribute_input_df, sv.attribute_binned_df, key='attributes_uploader', height=500)
        with model_col:
            util.ui_components.prepare_binned_df(workflow, sv.attribute_input_df, sv.attribute_binned_df, sv.attribute_subject_identifier)
            att_cols = [col for col in sv.attribute_binned_df.value.columns.values if col != 'Subject ID' and st.session_state[f'{workflow}_{col}'] == True]
            time_col = st.selectbox('Period column', [''] + [c for c in sv.attribute_binned_df.value.columns.values if c != 'Subject ID'])
            ready = len(att_cols) > 0 and time_col != ''

            if st.button("Add links to graph model", disabled=not ready):
                with st.spinner('Adding links to model...'):
                    df = sv.attribute_binned_df.value.copy(deep=True)
                    df['Entity ID'] = [str(x) for x in range(1, len(df) + 1)]
                    df['Entity ID'] = df['Entity ID'].astype(str)
                    pdf = df.copy(deep=True)[[time_col, 'Entity ID'] + att_cols]
                    pdf = pdf[pdf[time_col].notna() & pdf['Entity ID'].notna()]
                    pdf.rename(columns={time_col : 'Period'}, inplace=True)
                    
                    pdf['Period'] = pdf['Period'].astype(str)
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
