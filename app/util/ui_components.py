import streamlit as st
import pandas as pd
import numpy as np

import os
import sys

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

def single_csv_uploader(upload_label, last_uploaded_file_name_var, raw_df_var, processed_df_var, key, show_rows=1000, height=250):
    file = st.file_uploader(upload_label, type=['csv'], accept_multiple_files=False, key=key)
    if file != None and file.name != last_uploaded_file_name_var.value:
        last_uploaded_file_name_var.value = file.name
        df = pd.read_csv(file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        df = util.df_functions.fix_null_ints(df)
        raw_df_var.value = df
        processed_df_var.value = df.copy(deep=True)
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