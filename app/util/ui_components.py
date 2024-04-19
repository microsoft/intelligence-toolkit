# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import re
import streamlit as st
import pandas as pd
import numpy as np

import os
import math
import sys

from dateutil import parser as dateparser
from collections import defaultdict

from util.download_pdf import add_download_pdf
import util.AI_API
import util.df_functions


def dataframe_with_selections(df, selections, selection_col, label, key, height=250):
    df_with_selections = df.copy()
    values = []
    for val in df[selection_col].to_list():
        if val in selections:
            values.append(True)
        else:
            values.append(False)
    df_with_selections.insert(0, label, values)
    df_with_selections[label] = df_with_selections[label].astype(bool)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={label: st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        use_container_width=True,
        height=height
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df[label]]
    selected_rows = selected_rows.drop(label, axis=1)

    return selected_rows

def report_download_ui(report_var, name):
    if type(report_var) == str:
        if len(report_var) == 0:
            return
        report_data = report_var
        c1, c2 = st.columns([1, 1])
        spaced_name = name.replace('_', ' ')
        with c1:
            st.download_button(f'Download AI {spaced_name} as MD', data=report_data, file_name=f'{name}.md', mime='text/markdown')
        with c2:
            add_download_pdf(f'{name}.pdf', report_data, f'Download AI {spaced_name} as PDF')
    elif len(report_var.value) > 0:
        report_data = report_var.value
        c1, c2 = st.columns([1, 1])
        spaced_name = name.replace('_', ' ')
        with c1:
            st.download_button(f'Download AI {spaced_name} as MD', data=report_data, file_name=f'{name}.md', mime='text/markdown')
        with c2:
            add_download_pdf(f'{name}.pdf', report_data, f'Download AI {spaced_name} as PDF')

def generative_ai_component(system_prompt_var, variables):
    st.markdown('##### Generative AI instructions')
    with st.expander('Edit AI System Prompt (advanced)', expanded=True):
        instructions_text = st.text_area('Contents of System Prompt used to generate AI outputs.', value=system_prompt_var.value["user_prompt"], height=200)
        if system_prompt_var.value["user_prompt"] != instructions_text:
            system_prompt_var.value["user_prompt"] = instructions_text
            st.rerun()
        reset_prompt = st.button('Reset to default')
    
    st.warning('This app uses AI and may not be error-free. Please verify critical details independently.')
    
    full_prompt = ' '.join([
        system_prompt_var.value["report_prompt"],
        system_prompt_var.value["user_prompt"],
        system_prompt_var.value["safety_prompt"]
    ])

    messages = util.AI_API.prepare_messages_from_message(full_prompt, variables)
    tokens = util.AI_API.count_tokens_in_message_list(messages)
    b1, b2 = st.columns([1, 4])
    ratio = 100 * tokens/util.AI_API.max_input_tokens
    with b1:
        generate = st.button('Generate', disabled=ratio > 100)
    with b2:
        message = f'AI input uses {tokens}/{util.AI_API.max_input_tokens} ({round(ratio, 2)}%) of token limit'
        if ratio <= 100:
            st.info(message)
        else:
            st.warning(message)
    return generate, messages, reset_prompt

def generative_batch_ai_component(system_prompt_var, variables, batch_name, batch_val, batch_size):
    st.markdown('##### Generative AI instructions')
    with st.expander('Edit AI System Prompt (advanced)', expanded=True):
        instructions_text = st.text_area('Contents of System Prompt used to generate AI outputs.', value=system_prompt_var.value["user_prompt"], height=200)
        system_prompt_var.value["user_prompt"] = instructions_text
        reset_prompt = st.button('Reset to default')

    st.warning('This app uses AI and may not be error-free. Please verify critical details independently.')
    batch_offset = 0
    batch_count_raw = (len(batch_val) // batch_size)
    batch_count_remaining = (len(batch_val) % batch_size)
    batch_count = batch_count_raw + 1 if batch_count_remaining != 0 else batch_count_raw
    batch_messages = []

    full_prompt = ' '.join([
        system_prompt_var.value["report_prompt"],
        system_prompt_var.value["user_prompt"],
        system_prompt_var.value["safety_prompt"]
    ])
    for i in range(batch_count):
        batch = batch_val[batch_offset:min(batch_offset+batch_size, len(batch_val))]
        batch_offset += batch_size
        batch_variables = dict(variables)
        batch_variables[batch_name] = batch.to_csv()
        batch_messages.append(util.AI_API.prepare_messages_from_message(full_prompt, batch_variables))
    tokens = util.AI_API.count_tokens_in_message_list(batch_messages[0] if len(batch_messages) != 0 else [])
    b1, b2 = st.columns([1, 4])
    ratio = 100 * tokens/util.AI_API.max_input_tokens
    with b1:
        generate = st.button('Generate', disabled=ratio > 100)
    with b2:
        st.markdown(f'AI input uses {tokens}/{util.AI_API.max_input_tokens} ({round(ratio, 2)}%) of token limit')
    return generate, batch_messages, reset_prompt

file_options = ['unicode-escape', 'utf-8', 'utf-8-sig']
file_encoding_default = 'unicode-escape'
def single_csv_uploader(workflow, upload_label, last_uploaded_file_name_var, input_df_var, processed_df_var, final_df_var, key, show_rows=10000, height=250):
    file = st.file_uploader(upload_label, type=['csv'], accept_multiple_files=False, key=key)
    if f'{key}_encoding' not in st.session_state:
        st.session_state[f'{key}_encoding'] = file_encoding_default

    with st.expander('File options'):
        encoding = st.selectbox('File encoding', options=file_options, key=f'{key}_encoding_sb', index=file_options.index(st.session_state[f'{key}_encoding']))

        reload = st.button('Reload', key=f'{key}_reload')
    if file != None and (file.name != last_uploaded_file_name_var.value or reload):
        st.session_state[f'{key}_encoding'] = encoding
        last_uploaded_file_name_var.value = file.name
        df = pd.read_csv(file, encoding=encoding, encoding_errors='ignore', low_memory=False)
        # df.columns = df.columns.str.strip()
        # df = util.df_functions.fix_null_ints(df)
        input_df_var.value = df
        processed_df_var.value = pd.DataFrame()
        final_df_var.value = pd.DataFrame()
        if f'{workflow}_binned_df' in st.session_state.keys():
            del st.session_state[f'{workflow}_binned_df']
        st.rerun()
    options = []
    if input_df_var is not None:
        options += ['Raw']
    if processed_df_var is not None:
        options += ['Processing']
    if final_df_var is not None:
        options += ['Final']
    # dfo = st.radio('Select data table', options=options, index=0, horizontal=True, key=f'{workflow}_{upload_label}_data_table_select')
    option_tabs = st.tabs(options)
    for ix, tab in enumerate(option_tabs):
        with tab:
            if options[ix] == 'Raw':
                st.dataframe(input_df_var.value[:show_rows], hide_index=True, use_container_width=True, height=height)
            elif options[ix] == 'Processing':
                st.dataframe(processed_df_var.value[:show_rows], hide_index=True, use_container_width=True, height=height)
            if options[ix] == 'Final':
                st.dataframe(final_df_var.value[:show_rows], hide_index=True, use_container_width=True, height=height)
                st.download_button('Download final dataset', final_df_var.value.to_csv(index=False), file_name='final_dataset.csv', disabled=len(final_df_var.value) == 0)

def multi_csv_uploader(upload_label, uploaded_files_var, outputs_dir, key, max_rows_var=0, show_rows=1000, height=250):
    if f'{key}_encoding' not in st.session_state:
        st.session_state[f'{key}_encoding'] = file_encoding_default

    files = st.file_uploader(upload_label, type=['csv'], accept_multiple_files=True, key=key)

    if files != None:
        for file in files:
            if file.name not in uploaded_files_var.value:              
                uploaded_files_var.value.append(file.name)
    selected_file = st.selectbox("Select a file to process", options = ['']+uploaded_files_var.value if files else [])
    with st.expander('File options'):
        st.number_input('Maximum rows to process (0 = all)', min_value=0, step=1000, key=max_rows_var.key)
        encoding = st.selectbox('File encoding', options=file_options, key=f'{key}_encoding_db', index=file_options.index(st.session_state[f'{key}_encoding']))
        reload = st.button('Reload', key=f'{key}_reload')

    df = pd.DataFrame()
    if selected_file not in [None, ''] or reload:
        st.session_state[f'{key}_encoding'] = encoding
        for file in files:
            if file.name == selected_file:
                df = pd.read_csv(file, encoding=encoding, nrows=max_rows_var.value, encoding_errors='ignore', low_memory=False) if max_rows_var.value > 0 else pd.read_csv(file, encoding=encoding, encoding_errors='ignore', low_memory=False)
                break
        st.dataframe(df[:show_rows], hide_index=True, use_container_width=True, height=height)
    return selected_file, df

def prepare_input_df(workflow, input_df_var, processed_df_var, output_df_var, identifier_var):
    if f'{workflow}_last_identifier' not in st.session_state.keys():
        st.session_state[f'{workflow}_last_identifier'] = ''
    if f'{workflow}_last_attributes' not in st.session_state.keys():
        st.session_state[f'{workflow}_last_attributes'] = []
    if f'{workflow}_last_suppress_zeros' not in st.session_state.keys():
        st.session_state[f'{workflow}_last_suppress_zeros'] = False
    if f'{workflow}_binned_df' not in st.session_state.keys() or len(st.session_state[f'{workflow}_binned_df']) == 0:
        st.session_state[f'{workflow}_binned_df'] = input_df_var.value.copy(deep=True)
    if f'{workflow}_selected_binned_cols' not in st.session_state.keys():
        st.session_state[f'{workflow}_selected_binned_cols'] = []
    if f'{workflow}_selected_binned_size' not in st.session_state:
        st.session_state[f'{workflow}_selected_binned_size'] = 'Year'
    if f'{workflow}_selected_num_attr' not in st.session_state:
        st.session_state[f'{workflow}_selected_num_attr'] = []
    if f'{workflow}_selected_num_bins' not in st.session_state:
        st.session_state[f'{workflow}_selected_num_bins'] = 5
    if f'{workflow}_selected_trim_percent' not in st.session_state:
        st.session_state[f'{workflow}_selected_trim_percent'] = 0.05
    if f'{workflow}_selected_compound_cols' not in st.session_state:
        st.session_state[f'{workflow}_selected_compound_cols'] = []

    with st.expander('Set subject identifier', expanded=False):

        index_id = 1 if st.session_state[f'{workflow}_last_identifier'] and st.session_state[f'{workflow}_last_identifier'] != 'Subject ID' else 0
        identifier = st.radio('Subject identifier', index=index_id, options=['Row number', 'ID column'], help='Select row number if each row of data represents a distinct individual, otherwise select ID column to link multiple rows to the same individual via their ID.')
        if identifier == 'ID column':
            options = ['']+list(input_df_var.value.columns.values)
            identifier_col = st.selectbox('Select subject identifier column', options=options, index=options.index(st.session_state[f'{workflow}_last_identifier']) if st.session_state[f'{workflow}_last_identifier'] in options else None, help='Select the column that contains the unique identifier for each individual in the dataset. This column will be used to link multiple rows to the same individual. If the dataset does not contain a unique identifier, select Row number. If the dataset contains multiple columns that could be used as the identifier, select the appropriate column.')
            if identifier_col != '' and identifier_col != st.session_state[f'{workflow}_last_identifier']:
                st.session_state[f'{workflow}_last_identifier'] = identifier_col
                identifier_var.value = identifier_col
                if len(processed_df_var.value) == 0:
                    processed_df_var.index = input_df_var.value.index
                processed_df_var.value['Subject ID'] = list(input_df_var.value[identifier_col])
                st.rerun()
        else:
            identifier_var.value = 'Subject ID'
            processed_df_var.value['Subject ID'] = [i for i in range(len(input_df_var.value))]
            if 'Subject ID' != st.session_state[f'{workflow}_last_identifier']:
                st.session_state[f'{workflow}_last_identifier'] = 'Subject ID'
                st.rerun()

    with st.expander('Select attribute columns to include', expanded=False):
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button('Select all', use_container_width=True):
                for col in input_df_var.value.columns.values:
                    st.session_state[f'{workflow}_{col}'] = True
        with b2:
            if st.button('Deselect all', use_container_width=True):
                for col in input_df_var.value.columns.values:
                    st.session_state[f'{workflow}_{col}'] = False
        for col in input_df_var.value.columns.values:
            if f'{workflow}_{col}' not in st.session_state.keys():
                st.session_state[f'{workflow}_{col}'] = False
            if col != 'Subject ID':
                input = st.checkbox(col, key=f'{workflow}_{col}_input', value=st.session_state[f'{workflow}_{col}'])            
                st.session_state[f'{workflow}_{col}'] = input
    # set processed_df_var to input_df_var filtered by selected columns
    selected_cols = [col for col in input_df_var.value.columns.values if st.session_state[f'{workflow}_{col}'] == True]
    processed_df_var.value = processed_df_var.value[['Subject ID']].copy()
    for col in selected_cols:
        
        if col in st.session_state[f'{workflow}_binned_df'].columns.values:
            processed_df_var.value[col] = list(st.session_state[f'{workflow}_binned_df'][col])
            processed_df_var.value[col] = processed_df_var.value[col].replace('nan', '')

    for (cols, delim) in st.session_state[f'{workflow}_selected_compound_cols']:
        for col in cols:
            if col in selected_cols:
                # add each value as a separate column with a 1 if the value is present in the compound column and None otherwise
                values = processed_df_var.value[col].apply(lambda x: [y.strip() for y in x.split(delim)] if type(x) == str else [])
                unique_values = set([v for vals in values for v in vals])
                unique_values = [x for x in unique_values if x != '']
                for val in unique_values:
                    st.session_state[f'{workflow}_{val}'] = False
                    processed_df_var.value[val] = values.apply(lambda x: '1' if val in x and val != 'nan' else '')
                    # processed_df_var.value[col][col+'_'+val] = values.apply(lambda x: 1 if val in x and val != 'nan' else None)
                processed_df_var.value.drop(columns=[col], inplace=True)

    if selected_cols != st.session_state[f'{workflow}_last_attributes']:
        processed_df_var.value = util.df_functions.fix_null_ints(processed_df_var.value)
        st.session_state[f'{workflow}_last_attributes'] = selected_cols
        st.rerun()


    with st.expander('Quantize datetime attributes', expanded=False):
        # quantize numeric columns into bins
        binnable_cols = []
        for col in processed_df_var.value.columns:
            if col != 'Subject ID':
                binnable_cols.append(col)

        selected_date_cols = st.multiselect('Select datetime attribute to quantize', default=st.session_state[f'{workflow}_selected_binned_cols'], options=binnable_cols, help='Select the datetime columns you want to quantize. Quantizing datetime columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.')

        bin_size_options = ['Year', 'Half', 'Quarter', 'Month', 'Day']
        bin_size = st.radio('Select bin size', index=bin_size_options.index(st.session_state[f"{workflow}_selected_binned_size"]), options=bin_size_options, help='Select the bin size for the datetime columns you want to quantize. Quantizing datetime columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.')
        
        # num_bins = st.number_input('Number of bins', value=5, help='Number of bins to use for each column. If 0, no binning will be performed. Fewer bins makes it easier to synthesize data, but reduces the amount of information in the data. More bins makes it harder to synthesize data, but preserves more information in the data.')
        # trim_percent = st.number_input('Trim percent', value=0.05, help='Percent of values to trim from the top and bottom of each column before binning. This helps to reduce the impact of outliers on the binning process. For example, if trim percent is 0.05, the top and bottom 5% of values will be trimmed from each column before binning. If 0, no trimming will be performed.')
        if st.button('Quantize selected columns', key='quantize_date'):
            st.session_state[f"{workflow}_selected_binned_cols"] = selected_date_cols
            st.session_state[f"{workflow}_selected_binned_size"] = bin_size
        
            for col in selected_date_cols:
                func = None
                if bin_size == 'Year':
                    def convert(x):
                        # parse as datetime using dateutil
                        try:
                            dt = dateparser.parse(str(x))
                            return str(dt.year)
                        except:
                            return ''
                    func = convert
                elif bin_size == 'Half':
                    def convert(x):
                        try:
                            dt = dateparser.parse(str(x))
                            half = 'H1' if dt.month < 7 else 'H2'
                            return str(dt.year) + '-' + half
                        except:
                            return ''
                    func = convert
                elif bin_size == 'Quarter':
                    def convert(x):
                        try:
                            dt = dateparser.parse(str(x))
                            quarter = 'Q1' if dt.month < 4 else 'Q2' if dt.month < 7 else 'Q3' if dt.month < 10 else 'Q4'
                            return str(dt.year) + '-' + quarter
                        except:
                            return ''
                    func = convert
                elif bin_size == 'Month':
                    def convert(x):
                        try:
                            dt = dateparser.parse(str(x))
                            return str(dt.year) + '-' + str(dt.month).zfill(2)
                        except:
                            return ''
                    func = convert
                elif bin_size == 'Day':
                    def convert(x):
                        try:
                            dt = dateparser.parse(str(x))
                            return str(dt.year) + '-' + str(dt.month).zfill(2) + '-' + str(dt.day).zfill(2)
                        except:
                            return ''
                    func = convert
                st.session_state[f'{workflow}_binned_df'][col] = input_df_var.value[col].apply(func)
            st.session_state[f'{workflow}_last_attributes'] = [] # hack to force second rerun and show any changes from binning
            st.rerun()

    with st.expander('Quantize numeric attributes', expanded=False):
        # quantize numeric columns into bins
        binnable_cols = []
        for col in processed_df_var.value:
            if col != 'Subject ID' and processed_df_var.value[col].dtype in ['float64', 'int64', 'Int64']:
                binnable_cols.append(col)

        selected_binnable_cols = st.multiselect('Select numeric attributes to quantize', binnable_cols, default=st.session_state[f'{workflow}_selected_num_attr'], help='Select the numeric columns you want to quantize. Quantizing numeric columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.')
        num_bins = st.number_input('Number of bins', value=st.session_state[f'{workflow}_selected_num_bins'], help='Number of bins to use for each column. If 0, no binning will be performed. Fewer bins makes it easier to synthesize data, but reduces the amount of information in the data. More bins makes it harder to synthesize data, but preserves more information in the data.')
        trim_percent = st.number_input('Trim percent', value=st.session_state[f'{workflow}_selected_trim_percent'], help='Percent of values to trim from the top and bottom of each column before binning. This helps to reduce the impact of outliers on the binning process. For example, if trim percent is 0.05, the top and bottom 5% of values will be trimmed from each column before binning. If 0, no trimming will be performed.')
        
        if st.button('Quantize selected columns', key='quantize_numeric'):
            if num_bins == 0:
                for col in selected_binnable_cols:
                    st.session_state[f'{workflow}_binned_df'][col] = input_df_var.value[col]
            else:
                for col in selected_binnable_cols:
                    distinct_values = tuple(sorted(input_df_var.value[col].unique()))
                    if len(distinct_values) < 2:
                        continue
                    if distinct_values == tuple([0, 1]):
                        continue

                    sorted_values = sorted(input_df_var.value[col].values)
                    # first, calculate the top and bottom trim_percent values and apply top and bottom coding
                    top = min(len(sorted_values)-1, math.floor(len(sorted_values) * (1 - trim_percent)))
                    top_trim = sorted_values[top]
                    bottom_trim = sorted_values[math.floor(len(sorted_values) * trim_percent)]
                    processed_df_var.value.loc[input_df_var.value[col] > top_trim, col] = top_trim
                    processed_df_var.value.loc[input_df_var.value[col] < bottom_trim, col] = bottom_trim
                    target_bin_width = (top_trim - bottom_trim) / num_bins
                    # round target bin width to a multiple of N x 10^k for N = [1, 2, 2.5, 5]. N and k should be chosen to exceed the target bin width by the least positive amount
                    k = math.floor(math.log10(target_bin_width))
                    n_bin_sizes = {}
                    n_excess = {}
                    for N in [1, 2, 2.5, 5]:
                        n_bin_sizes[N] = N * pow(10, k)
                        if n_bin_sizes[N] < target_bin_width:
                            n_bin_sizes[N] = N * pow(10, k+1)
                        n_excess[N] = n_bin_sizes[N] - target_bin_width
                    # find the N that minimizes the excess
                    min_excess_n = sorted([(n, e) for n, e in n_excess.items()], key=lambda x: x[1])[0][0]
                    selected_bin_size = n_bin_sizes[min_excess_n]
                    # next, calculate the bin edges
                    
                    lower_bin_edge = (bottom_trim // selected_bin_size) * selected_bin_size
                    bin_edges = [lower_bin_edge * 0.999999999999]
                    last_bin = lower_bin_edge
                    while last_bin < top_trim:
                        last_bin += selected_bin_size
                        bin_edges.append(last_bin)
                    st.session_state[f'{workflow}_binned_df'][col] = input_df_var.value[col].copy(deep=True).astype('float64')
                    # finally, bin the values
                    values, bins = pd.cut(processed_df_var.value[col], bins=bin_edges, retbins=True, include_lowest=False)

                    results = ['' if type(v) == float else '(' + str(v.left) + '-' + str(v.right) + ']' for v in values]
                    # processed_df_var.value[col] = processed_df_var.value[col].astype('str')
                    st.session_state[f'{workflow}_binned_df'][col] = results

            st.session_state[f'{workflow}_selected_num_attr'] = selected_binnable_cols
            st.session_state[f'{workflow}_selected_num_bins'] = num_bins 
            st.session_state[f'{workflow}_selected_trim_percent'] = trim_percent
            st.session_state[f'{workflow}_last_attributes'] = [] # hack to force second rerun and show any changes from binning
            st.rerun() 

    with st.expander('Expand compound values', expanded=False):
        options = [x for x in processed_df_var.value.columns.values if x != 'Subject ID']
        selected_compound_cols = st.multiselect('Select compound columns to expand', options, help='Select the columns you want to expand into separate columns. If you do not select any columns, no expansion will be performed.')
        col_delimiter = st.text_input('Column delimiter', value='', help='The character used to separate values in compound columns. If the delimiter is not present in a cell, the cell will be left unchanged.')
        if st.button('Expand selected columns', key='expand_compound'):
            to_add = (selected_compound_cols, col_delimiter)
            if col_delimiter != '' and to_add not in st.session_state[f'{workflow}_selected_compound_cols']:
                st.session_state[f'{workflow}_selected_compound_cols'].append(to_add)
                st.rerun() 


    with st.expander('Suppress insignificant attribute values', expanded=False):
        if f'{workflow}_min_count' not in st.session_state.keys():
            st.session_state[f'{workflow}_min_count'] = 0
        min_value = st.number_input('Minimum value count', key=f'{workflow}_min_count_input', value=st.session_state[f'{workflow}_min_count'], help='Minimum count of an attribute value to be included in the sensitive dataset. If 0, no filtering will be performed.')
        st.session_state[f'{workflow}_min_count'] = min_value
        bdf = st.session_state[f'{workflow}_binned_df']
        for col in processed_df_var.value.columns:
            if col != 'Subject ID':
                if col not in bdf:
                    continue
                value_counts = bdf[col].value_counts()
                # convert to dict with value as key and count as value
                value_counts = dict(zip(value_counts.index, value_counts.values))

                # remove any values that are less than the minimum count
                if bdf[col].dtype == 'str':
                    print(f'Processing {col} as string')
                    bdf[col] = bdf[col].apply(lambda x: '' if x in value_counts and value_counts[x] < min_value else str(x))
                elif bdf[col].dtype == 'float64':
                    print(f'Processing {col} as float')
                    bdf[col] = bdf[col].apply(lambda x: np.nan if x in value_counts and value_counts[x] < min_value else x)
                elif bdf[col].dtype == 'int64':
                    print(f'Processing {col} as int')
                    bdf[col] = bdf[col].apply(lambda x: -sys.maxsize if x in value_counts and value_counts[x] < min_value else x)
                    bdf[col] = bdf[col].astype('Int64')
                    bdf[col] = bdf[col].replace(-sys.maxsize, np.nan)
                else:
                    print(f'Processing {col} as string')
                    bdf[col] = bdf[col].apply(lambda x: '' if x in value_counts and value_counts[x] < min_value else str(x))

        if f'{workflow}_suppress_zeros' not in st.session_state.keys():
            st.session_state[f'{workflow}_suppress_zeros'] = False
        suppress_zeros = st.checkbox('Suppress binary 0s', key=f'{workflow}_suppress_zeros_input', value=st.session_state[f'{workflow}_suppress_zeros'], help='For binary columns, maps the number 0 to None. This is useful when only the presence of an attribute is important, not the absence.')
        if suppress_zeros != st.session_state[f'{workflow}_suppress_zeros']:
            st.session_state[f'{workflow}_suppress_zeros'] = suppress_zeros
            if suppress_zeros:
                for col in bdf.columns.values:
                    if col != 'Entity ID' and len(bdf[col].unique()) <= 2:
                        if 0 in [x for x in bdf[col].unique()]:
                            bdf[col] = input_df_var.value[col].replace(0, np.nan)
                            processed_df_var.value[col] = bdf[col]
            else:
                for col in bdf.columns.values:
                    if col != 'Entity ID' and len(bdf[col].unique()) <= 2:
                        bdf[col] = input_df_var.value[col]
                        processed_df_var.value[col] = bdf[col]
            st.rerun()

    if st.button('Generate final dataset', disabled=len(processed_df_var.value.columns) < 2):
        with st.spinner('Transforming data...'):
            if identifier == 'ID column' and identifier_col != '':
                # Drop empty Subject ID rows
                filtered = processed_df_var.value.dropna(subset=['Subject ID'])
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
                if len(expanded_atts) > 0:
                    new_rows = []
                    for i, row in melted.iterrows():
                        if row['Attribute'] in expanded_atts:
                            if str(row['Value']) not in ['', '<NA>']:
                                new_rows.append([row['Subject ID'], row['Attribute']+'_'+str(row['Value']), '1'])
                        else:
                            new_rows.append([row['Subject ID'], row['Attribute'], str(row['Value'])])
                    melted = pd.DataFrame(new_rows, columns=['Subject ID', 'Attribute', 'Value'])
                    # convert back to wide format
                    wdf = melted.pivot(index='Subject ID', columns='Attribute', values='Value').reset_index()
                    # wdf = wdf.drop(columns=['Subject ID'])
                    
                    output_df_var.value = wdf
                else:
                    wdf = processed_df_var.value.copy(deep=True)
                    output_df_var.value = wdf
            else:
                wdf = processed_df_var.value.copy(deep=True)
                output_df_var.value = wdf
            output_df_var.value.replace({'<NA>': np.nan}, inplace=True)
            output_df_var.value.replace({'nan': ''}, inplace=True)
            output_df_var.value.replace({'1.0': '1'}, inplace=True)
            st.rerun()
    if len(output_df_var.value) > 0:
        st.success('Data preparation complete.')
    else:
        st.warning('Generate final dataset to continue.')

def validate_ai_report(messages, result, show_status = True):
    if show_status:
        st.status('Validating AI report and generating faithfulness score...', expanded=False, state='running')
    validation, messages_to_llm = util.AI_API.validate_report(messages, result)
    return re.sub(r"```json\n|\n```", "", validation), messages_to_llm