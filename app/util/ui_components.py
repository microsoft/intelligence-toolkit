import streamlit as st
import pandas as pd
import numpy as np

import os
import math
import sys

from dateutil import parser as dateparser

import util.AI_API
import util.df_functions

def dataframe_with_selections(df, height=250):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True, )},
        disabled=df.columns,
        use_container_width=True,
        height=height
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def generative_ai_component(system_prompt_var, instructions_var, variables):
    st.markdown('##### Generative AI instructions')
    with st.expander('Edit AI System Prompt (advanced)', expanded=False):
        st.text_area('Contents of System Prompt used to generate AI outputs. Do not edit {AI inputs} in curly brackets.', key=system_prompt_var.key, value=system_prompt_var.value, height=200)
    st.text_area('Instructions (optional - use to guide output)', key=instructions_var.key, value=instructions_var.value, height=100)
    variables['instructions'] = instructions_var.value
    messages = util.AI_API.prepare_messages_from_message(system_prompt_var.value, variables)
    tokens = util.AI_API.count_tokens_in_message_list(messages)
    b1, b2 = st.columns([1, 4])
    ratio = 100 * tokens/util.AI_API.max_input_tokens
    with b1:
        generate = st.button('Generate', disabled=ratio > 100)
    with b2:
        st.markdown(f'AI input uses {tokens}/{util.AI_API.max_input_tokens} ({round(ratio, 2)}%) of token limit')
    return generate, messages

def generative_batch_ai_component(system_prompt_var, instructions_var, variables, batch_name, batch_val, batch_size):
    st.markdown('##### Generative AI instructions')
    with st.expander('Edit AI System Prompt (advanced)', expanded=False):
        st.text_area('Contents of System Prompt used to generate AI outputs. Do not edit {AI inputs} in curly brackets.', key=system_prompt_var.key, value=system_prompt_var.value, height=200)
    st.text_area('Instructions (optional - use to guide output)', key=instructions_var.key, value=instructions_var.value, height=100)
    
    batch_offset = 0
    batch_count = (len(batch_val) // batch_size) + 1
    batch_messages = []
    variables['instructions'] = instructions_var.value
    for i in range(batch_count):
        batch = batch_val[batch_offset:min(batch_offset+batch_size, len(batch_val))]
        batch_offset += batch_size
        batch_variables = dict(variables)
        batch_variables[batch_name] = batch.to_csv()
        batch_messages.append(util.AI_API.prepare_messages_from_message(system_prompt_var.value, batch_variables))
    tokens = util.AI_API.count_tokens_in_message_list(batch_messages[0])
    b1, b2 = st.columns([1, 4])
    ratio = 100 * tokens/util.AI_API.max_input_tokens
    with b1:
        generate = st.button('Generate', disabled=ratio > 100)
    with b2:
        st.markdown(f'AI input uses {tokens}/{util.AI_API.max_input_tokens} ({round(ratio, 2)}%) of token limit')
    return generate, batch_messages

def single_csv_uploader(upload_label, last_uploaded_file_name_var, input_df_var, processed_df_var, key, show_rows=10000, height=250):
    file = st.file_uploader(upload_label, type=['csv'], accept_multiple_files=False, key=key)
    if file != None and file.name != last_uploaded_file_name_var.value:
        last_uploaded_file_name_var.value = file.name
        df = pd.read_csv(file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        df = util.df_functions.fix_null_ints(df)
        input_df_var.value = df
        processed_df_var.value = df.copy(deep=True)
        print('Resetting processed_df_var')
    if len(processed_df_var.value) > 0:
        st.dataframe(processed_df_var.value[:show_rows], hide_index=True, use_container_width=True, height=height)

def multi_csv_uploader(upload_label, uploaded_files_var, outputs_dir, key, max_rows_var=0, show_rows=1000, height=250):
    files = st.file_uploader(upload_label, type=['csv'], accept_multiple_files=True, key=key)
    st.number_input('Maximum rows to process (0 = all)', min_value=0, step=1000, key=max_rows_var.key, value=max_rows_var.value)
    if files != None:
        for file in files:
            if file.name not in uploaded_files_var.value:
                df = pd.read_csv(file, encoding='utf-8-sig')
                df.columns = df.columns.str.strip()
                df = util.df_functions.fix_null_ints(df)
                df.to_csv(os.path.join(outputs_dir, file.name), index=False)
                uploaded_files_var.value.append(file.name)
    selected_file = st.selectbox("Select a file to process", uploaded_files_var.value)
    
    df = pd.DataFrame()
    if selected_file != None:
        df = pd.read_csv(os.path.join(outputs_dir, selected_file))
        st.dataframe(df[:show_rows], hide_index=True, use_container_width=True, height=height)
    return selected_file, df

def prepare_binned_df(workflow, input_df_var, processed_df_var, identifier_var, min_count_var):
    # processed_df_var.value = input_df_var.value.copy(deep=True)
    with st.expander('Set subject identifier', expanded=True):
        identifier = st.radio('Subject identifier', options=['Row number', 'ID column'])
        if identifier == 'ID column':

            options = ['']+list(processed_df_var.value.columns.values)
            identifier = st.selectbox('Select subject identifier column', options=options)
            if identifier != '':
                if 'Subject ID' in processed_df_var.value.columns:
                    # remove
                    processed_df_var.value.drop(columns=['Subject ID'], inplace=True)
                identifier_var.value = identifier
                processed_df_var.value.rename(columns={identifier: 'Subject ID'}, inplace=True)
        else:
            processed_df_var.value['Subject ID'] = [i for i in range(len(processed_df_var.value))]
            identifier_var.value = 'Subject ID'
    

    with st.expander('Quantize datetime attributes', expanded=False):
        # quantize numeric columns into bins
        binnable_cols = []
        for col in input_df_var.value.columns:
            binnable_cols.append(col)

        selected_date_cols = st.multiselect('Select datetime attribute to quantize', binnable_cols, help='Select the datetime columns you want to quantize. Quantizing datetime columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.')
        bin_size = st.radio('Select bin size', options=['Year', 'Half', 'Quarter', 'Month', 'Day'], help='Select the bin size for the datetime columns you want to quantize. Quantizing datetime columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.')
        # num_bins = st.number_input('Number of bins', value=5, help='Number of bins to use for each column. If 0, no binning will be performed. Fewer bins makes it easier to synthesize data, but reduces the amount of information in the data. More bins makes it harder to synthesize data, but preserves more information in the data.')
        # trim_percent = st.number_input('Trim percent', value=0.05, help='Percent of values to trim from the top and bottom of each column before binning. This helps to reduce the impact of outliers on the binning process. For example, if trim percent is 0.05, the top and bottom 5% of values will be trimmed from each column before binning. If 0, no trimming will be performed.')
        if st.button('Quantize selected columns', key='quantize_date'):
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
                processed_df_var.value[col] = input_df_var.value[col].apply(func)
            st.rerun()

    with st.expander('Quantize numeric attributes', expanded=False):
        # quantize numeric columns into bins
        binnable_cols = []
        for col in input_df_var.value:
            if input_df_var.value[col].dtype in ['float64', 'int64', 'Int64']:
                binnable_cols.append(col)

        selected_binnable_cols = st.multiselect('Select numeric attributes to quantize', binnable_cols, help='Select the numeric columns you want to quantize. Quantizing numeric columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.')
        num_bins = st.number_input('Number of bins', value=5, help='Number of bins to use for each column. If 0, no binning will be performed. Fewer bins makes it easier to synthesize data, but reduces the amount of information in the data. More bins makes it harder to synthesize data, but preserves more information in the data.')
        trim_percent = st.number_input('Trim percent', value=0.05, help='Percent of values to trim from the top and bottom of each column before binning. This helps to reduce the impact of outliers on the binning process. For example, if trim percent is 0.05, the top and bottom 5% of values will be trimmed from each column before binning. If 0, no trimming will be performed.')
        if st.button('Quantize selected columns', key='quantize_numeric'):
            if num_bins == 0:
                processed_df_var.value = input_df_var.value.copy(deep=True)
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
                    processed_df_var.value[col] = input_df_var.value[col].copy(deep=True).astype('float64')
                    # finally, bin the values
                    values, bins = pd.cut(processed_df_var.value[col], bins=bin_edges, retbins=True, include_lowest=False)

                    results = ['' if type(v) == float else '(' + str(v.left) + '-' + str(v.right) + ']' for v in values]
                    # processed_df_var.value[col] = processed_df_var.value[col].astype('str')
                    processed_df_var.value[col] = results
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
                st.checkbox(col, key=f'{workflow}_{col}', value=st.session_state[f'{workflow}_{col}'])

    with st.expander('Prefilter rare attribute values', expanded=False):
        min_value = st.number_input('Minimum count', key=min_count_var.key, value=min_count_var.value, help='Minimum count of an attribute value to be included in the sensitive dataset. If 0, no filtering will be performed.')
        for col in processed_df_var.value.columns:
            if col != 'Subject ID':
                value_counts = processed_df_var.value[col].value_counts()
                # convert to dict with value as key and count as value
                value_counts = dict(zip(value_counts.index, value_counts.values))

                # remove any values that are less than the minimum count
                if processed_df_var.value[col].dtype == 'str':
                    processed_df_var.value[col] = processed_df_var.value[col].apply(lambda x: '' if x in value_counts and value_counts[x] < min_value else str(x))
                elif processed_df_var.value[col].dtype == 'float64':
                    processed_df_var.value[col] = processed_df_var.value[col].apply(lambda x: np.nan if x in value_counts and value_counts[x] < min_value else x)
                elif processed_df_var.value[col].dtype == 'int64':
                    processed_df_var.value[col] = processed_df_var.value[col].apply(lambda x: -sys.maxsize if x in value_counts and value_counts[x] < min_value else x)
                    processed_df_var.value[col] = processed_df_var.value[col].astype('Int64')
                    processed_df_var.value[col] = processed_df_var.value[col].replace(-sys.maxsize, np.nan)
