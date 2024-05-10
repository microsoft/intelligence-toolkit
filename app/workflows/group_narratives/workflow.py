# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
import pandas as pd
import streamlit as st
import util.df_functions
import workflows.group_narratives.prompts as prompts
import workflows.group_narratives.variables as vars
from util import ui_components

def get_intro():
    file_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(file_path, 'r') as file:
        return file.read()

def create(sv: vars.SessionVariables, workflow = None):
    intro_tab, prepare_tab, summarize_tab, generate_tab = st.tabs(['Group narratives workflow:', 'Upload data to narrate', 'Prepare data summary', 'Generate AI group reports',])

    with intro_tab:
        st.markdown(get_intro())
    with prepare_tab:
        uploader_col, model_col = st.columns([1, 1])
        with uploader_col:
            ui_components.single_csv_uploader(workflow, 'Upload CSV to narrate', sv.narrative_last_file_name, sv.narrative_input_df, sv.narrative_binned_df, sv.narrative_final_df, uploader_key=sv.narrative_upload_key.value,key='narrative_uploader', height=400)
        with model_col:
            ui_components.prepare_input_df(workflow, sv.narrative_input_df, sv.narrative_binned_df, sv.narrative_final_df, sv.narrative_subject_identifier)
            sv.narrative_final_df.value = util.df_functions.fix_null_ints(sv.narrative_final_df.value)
            sv.narrative_final_df.value = sv.narrative_final_df.value.astype(str).replace('<NA>', '').replace('nan', '')
        
    with summarize_tab:
        if len(sv.narrative_final_df.value) == 0:
            st.warning('Upload data to continue.')
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown('##### Define summary model')
                sorted_atts = []
                sorted_cols = sorted(sv.narrative_final_df.value.columns)
                for col in sorted_cols:
                    if col == 'Subject ID':
                        continue
                    vals = [f'{col}:{x}' for x in sorted(sv.narrative_final_df.value[col].astype(str).unique()) if x not in ['', '<NA>', 'nan', 'NaN', 'None', 'none', 'NULL', 'null']]
                    sorted_atts.extend(vals)
                
                filters = st.multiselect('After filtering to records matching these values:', sorted_atts, default=sv.narrative_filters.value)
                groups = st.multiselect('Compare groups of records with different combinations of these attributes:', sorted_cols, default=sv.narrative_groups.value)
                aggregates = st.multiselect('Using counts of these attributes:', sorted_cols, default=sv.narrative_aggregates.value)
                temporal_options = [''] + sorted_cols
                temporal = st.selectbox('Across windows of this temporal/ordinal attribute (optional):', temporal_options, index=temporal_options.index(sv.narrative_temporal.value))

                model = st.button('Create summary', disabled=len(groups) == 0 or len(aggregates) == 0)
            
            with c2:
                st.markdown('##### Data summary')
                if model:
                    sv.narrative_filters.value = filters
                    sv.narrative_groups.value = groups
                    sv.narrative_aggregates.value = aggregates
                    sv.narrative_temporal.value = temporal

                    sv.narrative_model_df.value = sv.narrative_final_df.value.copy(deep=True)
                    sv.narrative_model_df.value['Subject ID'] = [str(x) for x in range(1, len(sv.narrative_model_df.value) + 1)]

                    sv.narrative_model_df.value = sv.narrative_model_df.value.replace('', None)

                    # wide df for model
                    wdf = sv.narrative_model_df.value
                    print(wdf)
                    initial_row_count = len(wdf)
                    
                    if len(filters) > 0:
                        for f in filters:
                            col, val = f.split(':')
                            wdf = wdf[wdf[col] == val]
                    filtered_row_count = len(wdf)
                    dataset_proportion = int(round(100 * filtered_row_count / initial_row_count if initial_row_count > 0 else 0, 0))
                    # narrow df for model
                    id_vars = groups + [temporal] if temporal != '' else groups
                    ndf = wdf.melt(id_vars=id_vars, value_vars=aggregates, var_name='Attribute', value_name='Value')
                    ndf.dropna(subset=['Value'], inplace=True)
                    ndf['Attribute Value'] = ndf.apply(lambda x : str(x['Attribute']) +  ':' + str(x['Value']), axis=1)
                    temporal_atts = []
                    
                    # create group df
                    gdf = wdf.melt(id_vars=groups, value_vars=['Subject ID'], var_name='Attribute', value_name='Value')
                    
                    gdf['Attribute Value'] = gdf['Attribute'] + ':' + gdf['Value']
                    gdf = gdf.groupby(groups).size().reset_index(name='Group Count')
                    # Add group ranks
                    gdf['Group Rank'] = gdf['Group Count'].rank(ascending=False, method='max', na_option='bottom')

                    # create attribute df 
                    adf = ndf.groupby(groups + ['Attribute Value']).size().reset_index(name='Attribute Count')
                    # Ensure all groups have entries for all attribute values
                    for name, group in adf.groupby(groups):
                        for att_val in adf['Attribute Value'].unique():
                            # count rows with this group and attribute value
                            row_count = len(group[group['Attribute Value'] == att_val])
                            if row_count == 0:
                                adf.loc[len(adf)] = [*name, att_val, 0]

                    for att_val in adf['Attribute Value'].unique():
                        adf.loc[adf['Attribute Value'] == att_val, 'Attribute Rank'] = adf[adf['Attribute Value'] == att_val]['Attribute Count'].rank(ascending=False, method='max', na_option='bottom')

                    ldf = None

                    # create Window df 
                    if temporal != '':
                        temporal_atts = sorted(sv.narrative_model_df.value[temporal].astype(str).unique())
                        ldf = wdf.melt(id_vars=groups + [temporal], value_vars=aggregates, var_name='Attribute', value_name='Value')
                        ldf['Attribute Value'] = ldf['Attribute'] + ':' + ldf['Value']
                        # group by groups and count attribute values
                        ldf = ldf.groupby(groups + [temporal, 'Attribute Value']).size().reset_index(name=f'{temporal} Window Count')


                    tdfs = []
                    if len(temporal_atts) > 0:
                        # Add in 0 counts for any missing temporal attribute values across all groups and attribute values
                        for name, group in ldf.groupby(groups):
                            for att_val in ldf['Attribute Value'].unique():
                                for time_val in ldf[temporal].unique():
                                    if len(group[(group[temporal] == time_val) & (group[f'{temporal} Window Count'] == att_val)]) == 0:
                                        ldf.loc[len(ldf)] = [*name, time_val, att_val, 0]

                        # Calculate deltas in counts within each group and attribute value
                        for name, ddf in ldf.groupby(groups + ['Attribute Value']):
                            ldf.loc[ddf.index, f'{temporal} Window Delta'] = ddf[f'{temporal} Window Count'].diff().fillna(0)
                        for tatt in temporal_atts:
                            tdf = ldf[ldf[temporal] == tatt].copy(deep=True)
                            # rank counts for each attribute value
                            for att_val in tdf['Attribute Value'].unique():
                                tdf.loc[(tdf['Attribute Value'] == att_val), f'{temporal} Window Rank'] = tdf[tdf['Attribute Value'] == att_val][f'{temporal} Window Count'].rank(ascending=False, method='first')
                            tdfs.append(tdf)
                        ldf = pd.concat(tdfs).sort_values(by=temporal)

                    # Create overall df
                    odf = ldf.merge(gdf, on=[*groups], how='left', suffixes=['', '_r']) if temporal != '' else adf.merge(gdf, on=[*groups], how='left', suffixes=['', '_r'])
                    odf = odf.merge(adf, on=[*groups, 'Attribute Value'], how='left', suffixes=['', '_r'])
                    odf = odf.sort_values(by=[*groups], ascending=True)
                    if temporal != '':
                        odf.rename(columns={temporal: f'{temporal} Window'}, inplace=True)
                        odf[f'{temporal} Window Rank'] = odf[f'{temporal} Window Rank'].astype(int)
                        odf[f'{temporal} Window Delta'] = odf[f'{temporal} Window Delta'].astype(int)
                    odf['Attribute Rank'] = odf['Attribute Rank'].astype(int)
                    odf['Group Rank'] = odf['Group Rank'].astype(int)
                    
                    sv.narrative_model_df.value = odf[[*groups, 'Group Count', 'Group Rank', 'Attribute Value', 'Attribute Count', 'Attribute Rank', f'{temporal} Window', f'{temporal} Window Count', f'{temporal} Window Rank', f'{temporal} Window Delta']] if temporal != '' else odf[[*groups, 'Group Count', 'Group Rank', 'Attribute Value', 'Attribute Count', 'Attribute Rank']]
                    groups_text = '['+ ', '.join(['**'+g+'**' for g in groups]) + ']'
                    filters_text = '['+ ', '.join(['**'+f.replace(':', '\\:')+'**' for f in filters]) + ']'
                    description = 'This table shows:'
                    description += f'\n- A summary of **{filtered_row_count}** data records matching {filters_text}, representing **{dataset_proportion}%** of the overall dataset' if len(filters) > 0 else f'\n- A summary of all **{initial_row_count}** data records'   
                    description += f'\n- The **Group Count** of records for all {groups_text} groups, and corresponding **Group Rank**'
                    description += f'\n- The **Attribute Count** of each **Attribute Value** for all {groups_text} groups, and corresponding **Attribute Rank**'
                    if temporal != '':
                        description += f'\n- The **{temporal} Window Count** of each **Attribute Value** for each **{temporal} Window** for all {groups_text} groups, and corresponding **{temporal} Window Rank**'
                        description += f'\n- The **{temporal} Window Delta**, or change in the **Attribute Value Count** for successive **{temporal} Window** values, within each {groups_text} group'
                    sv.narrative_description.value = description
                    st.rerun()
                if len(sv.narrative_model_df.value) > 0:
                    st.dataframe(sv.narrative_model_df.value, hide_index=True, use_container_width=True, height=500)
                    
                    st.markdown(sv.narrative_description.value)
                    st.download_button('Download data', data=sv.narrative_model_df.value.to_csv(index=False, encoding='utf-8-sig'), file_name='narrative_data_summary.csv', mime='text/csv')
                

    with generate_tab:
        if len(sv.narrative_model_df.value) == 0:
            st.warning('Prepare data summary to continue.')
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('##### Data summary filters')
                groups = sorted(sv.narrative_model_df.value.groupby(sv.narrative_groups.value).groups.keys())
                b1, b2 = st.columns([1, 1])
                with b1:
                    selected_groups = st.multiselect('Select specific groups to narrate:', list(groups), default=sv.narrative_selected_groups.value)
                with b2:
                    top_group_ranks = st.number_input('OR Select top group ranks to narrate:', min_value=0, max_value=9999999999, value=sv.narrative_top_groups.value)
                fdf = sv.narrative_model_df.value.copy(deep=True)
                filter_description = ''
                if len(selected_groups) > 0:
                    fdf = fdf[fdf.set_index(sv.narrative_groups.value).index.isin(selected_groups)]
                    filter_description = f'Filtered to the following groups only: {", ".join([str(s) for s in selected_groups])}'
                elif top_group_ranks > 0:
                    fdf = fdf[fdf['Group Rank'] <= top_group_ranks]
                    filter_description = f'Filtered to the top {top_group_ranks} groups by record count'
                num_rows = len(fdf)   
                st.markdown(f'##### Filtered data summary to narrate ({num_rows} rows)')
                st.dataframe(fdf, hide_index=True, use_container_width=True, height=280)
                variables = {
                    'description': sv.narrative_description.value,
                    'dataset': fdf.to_csv(index=False, encoding='utf-8-sig'),
                    'filters': filter_description
                }
                generate, messages, reset = ui_components.generative_ai_component(sv.narrative_system_prompt, variables)
                if reset:
                    sv.narrative_system_prompt.value["user_prompt"] = prompts.user_prompt
                    st.rerun()
            with c2:
                st.markdown('##### Data narrative')
                
                narrative_placeholder = st.empty()
                gen_placeholder = st.empty()
                if generate:
                    sv.narrative_selected_groups.value = selected_groups
                    sv.narrative_top_groups.value = top_group_ranks

                    on_callback = ui_components.create_markdown_callback(narrative_placeholder)
                    result = ui_components.generate_text(messages, callbacks=[on_callback])
                    sv.narrative_report.value = result

                    validation, messages_to_llm = ui_components.validate_ai_report(messages, result)
                    sv.narrative_report_validation.value = validation
                    sv.narrative_report_validation_messages.value = messages_to_llm
                    st.rerun()
                else:
                    if sv.narrative_report.value == '':
                        gen_placeholder.warning('Press the Generate button to create an AI report for the selected groups.')
                narrative_placeholder.markdown(sv.narrative_report.value)
                
                ui_components.report_download_ui(sv.narrative_report, 'group_report')

                ui_components.build_validation_ui(sv.narrative_report_validation.value, sv.narrative_report_validation_messages.value, sv.narrative_report.value, workflow)