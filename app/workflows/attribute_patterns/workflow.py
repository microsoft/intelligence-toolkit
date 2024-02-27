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

import workflows.attribute_patterns.functions as functions
import workflows.attribute_patterns.classes as classes
import workflows.attribute_patterns.config as config
import workflows.attribute_patterns.prompts as prompts
import workflows.attribute_patterns.variables as vars

import workflows.attribute_patterns.config as config

import util.AI_API
import util.ui_components

def create():
    workflow = 'attribute_patterns'
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Attribute Patterns')
    sv = vars.SessionVariables('attribute_patterns')
    uploader_tab, detect_tab, explain_tab = st.tabs(['Create graph model', 'Detect patterns', 'Explain pattern'])
    df = None
    with uploader_tab:
        uploader_col, model_col = st.columns([2, 1])
        with uploader_col:
            util.ui_components.single_csv_uploader(workflow, 'Upload CSV', sv.attribute_last_file_name, sv.attribute_input_df, sv.attribute_binned_df, sv.attribute_final_df, key='attributes_uploader', height=500)
        with model_col:
            util.ui_components.prepare_input_df(workflow, sv.attribute_input_df, sv.attribute_binned_df, sv.attribute_final_df, sv.attribute_subject_identifier)
            options = [''] + [c for c in sv.attribute_final_df.value.columns.values if c != 'Subject ID']
            sv.attribute_time_col.value = st.selectbox('Period column', options, index=options.index(sv.attribute_time_col.value) if sv.attribute_time_col.value in options else 0)
            time_col = sv.attribute_time_col.value
            att_cols = [col for col in sv.attribute_final_df.value.columns.values if col not in ['Subject ID', time_col] and st.session_state[f'{workflow}_{col}'] == True]
            
            ready = len(att_cols) > 0 and sv.attribute_time_col.value != ''

            if st.button("Generate graph model", disabled=not ready):
                with st.spinner('Adding links to model...'):
                    time_col = sv.attribute_time_col.value
                    df = sv.attribute_final_df.value.copy(deep=True)
                    df['Subject ID'] = [str(x) for x in range(1, len(df) + 1)]
                    df['Subject ID'] = df['Subject ID'].astype(str)
                    pdf = df.copy(deep=True)[[time_col, 'Subject ID'] + att_cols]
                    pdf = pdf[pdf[time_col].notna() & pdf['Subject ID'].notna()]
                    pdf.rename(columns={time_col : 'Period'}, inplace=True)
                    
                    pdf['Period'] = pdf['Period'].astype(str)
                    pdf = pd.melt(pdf, id_vars=['Subject ID', 'Period'], value_vars=att_cols, var_name='Attribute Type', value_name='Attribute Value')
                    pdf = pdf[pdf['Attribute Value'] != '']
                    pdf['Full Attribute'] = pdf.apply(lambda x: str(x['Attribute Type']) + config.type_val_sep + str(x['Attribute Value']), axis=1)
                sv.attribute_dynamic_df.value = pdf
            if ready and len(sv.attribute_dynamic_df.value) > 0:
                st.markdown(f'Graph model has **{len(sv.attribute_dynamic_df.value)}** links spanning **{len(sv.attribute_dynamic_df.value["Subject ID"].unique())}** cases, **{len(sv.attribute_dynamic_df.value["Full Attribute"].unique())}** attributes, and **{len(sv.attribute_dynamic_df.value["Period"].unique())}** periods.')

    with detect_tab:
        if not ready or len(sv.attribute_final_df.value) == 0:
            st.markdown('Generate a graph model to continue.')
        else:
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown('##### Pattern detection')
                b1, b2 = st.columns([1, 1])
                with b1:
                    st.number_input('Minimum pattern count', min_value=1, step=1, key=sv.attribute_min_pattern_count.key, value=sv.attribute_min_pattern_count.value)
                with b2:
                    st.number_input('Maximum pattern length', min_value=1, step=1, key=sv.attribute_max_pattern_length.key, value=sv.attribute_max_pattern_length.value)
                if st.button('Detect patterns'):
                    with st.spinner('Detecting patterns...'):
                        sv.attribute_df.value, time_to_graph = functions.prepare_graph(sv)
                        
                        sv.attribute_embedding_df.value, sv.attribute_node_to_centroid.value, sv.attribute_period_embeddings.value = functions.generate_embedding(sv, sv.attribute_df.value, time_to_graph)
                            
                        rc = classes.RecordCounter(sv.attribute_dynamic_df.value)
                        sv.attribute_record_counter.value = rc
                        sv.attribute_pattern_df.value, sv.attribute_close_pairs.value, sv.attribute_all_pairs.value = functions.detect_patterns(sv)
            with c2:
                st.markdown('##### Detected patterns')
                if len(sv.attribute_pattern_df.value) == 0:
                    st.markdown('Detect patterns to proceed.')
                else:
                    prop = sv.attribute_close_pairs.value / sv.attribute_all_pairs.value if sv.attribute_all_pairs.value > 0 else 0
                    prop = round(prop, -int(np.floor(np.log10(abs(prop))))) if prop > 0 else 0
                    period_count = len(sv.attribute_pattern_df.value["period"].unique())
                    pattern_count = len(sv.attribute_pattern_df.value)
                    unique_count = len(sv.attribute_pattern_df.value['pattern'].unique())
                    st.markdown(f'Over **{period_count}** periods, detected **{pattern_count}** attribute patterns (**{unique_count}** unique).')
                    show_df = sv.attribute_pattern_df.value
                    tdf = functions.create_time_series_df(sv.attribute_record_counter.value, sv.attribute_pattern_df.value)
                    gb = GridOptionsBuilder.from_dataframe(show_df)
                    gb.configure_default_column(wrapText=True, enablePivot=False, enableValue=False, enableRowGroup=False)
                    gb.configure_selection(selection_mode="single", use_checkbox=False)
                    gb.configure_side_bar()
                    gridoptions = gb.build()

                    response = AgGrid(
                        show_df,
                        key='report_grid',
                        height=400,
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


                    if selected_pattern != '' and selected_pattern != sv.attribute_selected_pattern.value:
                        sv.attribute_selected_pattern.value = selected_pattern
                        sv.attribute_selected_pattern_period.value = selected_pattern_period
                        sv.attribute_report.value = ''
                        st.markdown('**Selected pattern: ' + selected_pattern + ' (' + selected_pattern_period + ')**')
                        tdf = tdf[tdf['pattern'] == selected_pattern]
                        sv.attribute_selected_pattern_df.value = tdf
                        sv.attribute_selected_pattern_att_counts.value = functions.compute_attribute_counts(sv.attribute_final_df.value, selected_pattern, time_col, selected_pattern_period)
                        count_ct = alt.Chart(tdf).mark_line().encode(
                            x='period:O',
                            y='count:Q',
                            color=alt.ColorValue('blue')
                        ).properties(
                            height = 200,
                            width = 600
                        )
                        st.altair_chart(count_ct, use_container_width=True)
    with explain_tab:
        if not ready or len(sv.attribute_final_df.value) == 0 or sv.attribute_selected_pattern.value == '':
            st.markdown('Select a pattern to continue.')
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                variables = {
                    'pattern': sv.attribute_selected_pattern.value,
                    'period': sv.attribute_selected_pattern_period.value,
                    'time_series': sv.attribute_selected_pattern_df.value.to_csv(index=False),
                    'attribute_counts': sv.attribute_selected_pattern_att_counts.value.to_csv(index=False)
                }
                generate, messages = util.ui_components.generative_ai_component(sv.attribute_system_prompt, sv.attribute_instructions, variables)
            with c2:
                st.markdown('##### Selected attribute pattern')
                selected_pattern = sv.attribute_selected_pattern.value
                selected_pattern_period = sv.attribute_selected_pattern_period.value
                if selected_pattern != '':
                    
                    st.markdown('**' + selected_pattern + ' (' + selected_pattern_period + ')**')
                    tdf = sv.attribute_selected_pattern_df.value

                    count_ct = alt.Chart(tdf).mark_line().encode(
                        x='period:O',
                        y='count:Q',
                        color=alt.ColorValue('blue')
                    ).properties(
                        height = 200,
                        width = 600
                    )
                    st.altair_chart(count_ct, use_container_width=True)
                report_placeholder = st.empty()
            
                if generate:
                    result = util.AI_API.generate_text_from_message_list(
                        placeholder=report_placeholder,
                        messages=messages,
                        prefix=''
                    )
                    sv.attribute_report.value = result
                report_placeholder.markdown(sv.attribute_report.value)
                st.download_button('Download pattern report', data=sv.attribute_report.value, file_name='attribute_pattern_report.md', mime='text/markdown', disabled=sv.attribute_report.value == '')
