# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import json
import os
import random
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import intelligence_toolkit.AI.utils as utils
from app.util.constants import FILE_ENCODING_DEFAULT, FILE_ENCODING_OPTIONS
from app.util.df_functions import get_current_time, quantize_datetime, quantize_numeric
from app.util.download_pdf import add_download_pdf
from app.util.enums import Mode
from app.util.openai_wrapper import UIOpenAIConfiguration
from intelligence_toolkit.AI.classes import LLMCallback
from intelligence_toolkit.AI.client import OpenAIClient
from intelligence_toolkit.AI.defaults import DEFAULT_MAX_INPUT_TOKENS
from intelligence_toolkit.helpers import df_functions
from intelligence_toolkit.helpers.texts import clean_for_column_name


def return_token_count(text: str) -> int:
    ai_configuration = UIOpenAIConfiguration().get_configuration()
    return utils.get_token_count(text, None, ai_configuration.model)


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
        height=height,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df[label]]
    return selected_rows.drop(label, axis=1)


def report_download_ui(report_var, name) -> None:
    if type(report_var) == str:
        if len(report_var) == 0:
            return
        report_data = report_var
        c1, c2 = st.columns([1, 1])
        spaced_name = name.replace("_", " ")
        with c1:
            st.download_button(
                f"Download AI {spaced_name} as MD",
                data=report_data,
                file_name=f"{name}.md",
                mime="text/markdown",
            )
        with c2:
            add_download_pdf(
                f"{name}.pdf", report_data, f"Download AI {spaced_name} as PDF"
            )
    elif len(report_var.value) > 0:
        report_data = report_var.value
        c1, c2 = st.columns([1, 1])
        spaced_name = name.replace("_", " ")
        with c1:
            st.download_button(
                f"Download AI {spaced_name} as MD",
                data=report_data,
                file_name=f"{name}.md",
                mime="text/markdown",
            )
        with c2:
            add_download_pdf(
                f"{name}.pdf", report_data, f"Download AI {spaced_name} as PDF"
            )


def generative_ai_component(
    system_prompt_var, variables
) -> tuple[bool, list[dict[str, str]], bool]:
    st.markdown("##### Generative AI instructions")

    with st.expander(
        "Edit AI system prompt used to generate output report", expanded=True
    ):
        reset_prompt = st.button("Discard prompt text changes")
        instructions_text = st.text_area(
            "Prompt text", value=system_prompt_var.value["user_prompt"], height=200
        )

    st.warning(
        "AI outputs may contain errors. Please verify details independently."
    )

    messages = utils.generate_messages(
        instructions_text,
        system_prompt_var.value["report_prompt"],
        variables,
        system_prompt_var.value["safety_prompt"],
    )
    tokens = return_token_count(messages)
    b1, b2 = st.columns([1, 4])
    ratio = 100 * tokens / DEFAULT_MAX_INPUT_TOKENS
    with b1:
        generate = st.button("Generate", disabled=ratio > 100)
        if generate:
            system_prompt_var.value["user_prompt"] = instructions_text

    with b2:
        message = f"AI input uses **{round(ratio, 2)}%** ({tokens}/{DEFAULT_MAX_INPUT_TOKENS}) of token limit"
        if ratio <= 100:
            st.info(message)
        else:
            st.error(message)
    return generate, messages, reset_prompt


def generative_batch_ai_component(
    system_prompt_var, variables, batch_name, batch_val, batch_size
) -> tuple[bool, list, bool]:
    st.markdown("##### Generative AI instructions")
    with st.expander("Edit AI System Prompt", expanded=True):
        reset_prompt = st.button("Discard prompt text changes")
        instructions_text = st.text_area(
            "Contents of System Prompt used to generate AI outputs.",
            value=system_prompt_var.value["user_prompt"],
            height=200,
        )

    st.warning(
        "AI outputs may contain errors. Please verify details independently."
    )
    batch_offset = 0
    batch_count_raw = len(batch_val) // batch_size
    batch_count_remaining = len(batch_val) % batch_size
    batch_count = batch_count_raw + 1 if batch_count_remaining != 0 else batch_count_raw
    batch_messages = []

    full_prompt = " ".join(
        [
            system_prompt_var.value["report_prompt"],
            instructions_text,
            system_prompt_var.value["safety_prompt"],
        ]
    )
    for _i in range(batch_count):
        batch = batch_val[batch_offset : min(batch_offset + batch_size, len(batch_val))]
        batch_offset += batch_size
        batch_variables = dict(variables)
        batch_variables[batch_name] = batch.to_csv()
        batch_messages.append(utils.prepare_messages(full_prompt, batch_variables))
    tokens = return_token_count(batch_messages[0] if len(batch_messages) != 0 else [])
    b1, b2 = st.columns([1, 4])
    ratio = 100 * tokens / DEFAULT_MAX_INPUT_TOKENS
    with b1:
        generate = st.button("Generate", disabled=ratio > 100)
        if generate:
            system_prompt_var.value["user_prompt"] = instructions_text
    with b2:
        st.markdown(
            f"AI input uses {tokens}/{DEFAULT_MAX_INPUT_TOKENS} ({round(ratio, 2)}%) of token limit"
        )
    return generate, batch_messages, reset_prompt

def single_csv_uploader(
    workflow,
    upload_label,
    last_uploaded_file_name_var,
    input_df_var,
    processed_df_var,
    key,
    show_rows=10000,
    height=250,
) -> None:
    if f"{workflow}_uploader_index" not in st.session_state:
        st.session_state[f"{workflow}_uploader_index"] = str(random.randint(0, 100))
    file = st.file_uploader(
        upload_label,
        type=["csv"],
        accept_multiple_files=False,
        key=key + "_file_uploader_" + st.session_state[f"{workflow}_uploader_index"],
    )
    if f"{key}_encoding" not in st.session_state:
        st.session_state[f"{key}_encoding"] = FILE_ENCODING_DEFAULT

    col1, col2 = st.columns([1, 2])
    with col1:
        encoding = st.selectbox(
            "File encoding",
            disabled=file is None,
            options=FILE_ENCODING_OPTIONS,
            key=f"{key}_encoding_sb",
            index=FILE_ENCODING_OPTIONS.index(st.session_state[f"{key}_encoding"]),
        )
    with col2:
        st.text("")
        st.text("")
        reload = st.button("Reload", key=f"{key}_reload", disabled=file is None)

    if file is not None and (file.name != last_uploaded_file_name_var.value or reload):
        st.session_state[f"{key}_encoding"] = encoding
        last_uploaded_file_name_var.value = file.name
        df = pd.read_csv(
            file, encoding=encoding, encoding_errors="ignore", low_memory=False
        )
        df.columns = [clean_for_column_name(col) for col in df.columns]
        input_df_var.value = df
        processed_df_var.value = pd.DataFrame()
        if f"{workflow}_intermediate_dfs" in st.session_state:
            st.session_state[f"{workflow}_intermediate_dfs"].clear()
        st.rerun()
    options = []
    if input_df_var is not None:
        options += ["Input data"]
    if processed_df_var is not None:
        options += ["Prepared data"]
    option_tabs = st.tabs(options)
    for ix, tab in enumerate(option_tabs):
        with tab:
            if options[ix] == "Input data":
                st.dataframe(
                    input_df_var.value[:show_rows],
                    hide_index=True,
                    use_container_width=True,
                    height=height,
                )
            elif options[ix] == "Prepared data":
                st.dataframe(
                    processed_df_var.value[:show_rows],
                    hide_index=True,
                    use_container_width=True,
                    height=height,
                )
                st.download_button(
                    "Download prepared dataset",
                    processed_df_var.value.to_csv(index=False),
                    file_name="prepared_dataset.csv",
                    disabled=len(processed_df_var.value) == 0,
                )


def multi_csv_uploader(
    upload_label,
    uploaded_files_var,
    key,
    max_rows_var=0,
    show_rows=1000,
    height=250,
) -> tuple[str | Any | None, pd.DataFrame]:
    if f"{key}_encoding" not in st.session_state:
        st.session_state[f"{key}_encoding"] = FILE_ENCODING_DEFAULT

    if f"{key}_uploader_index" not in st.session_state:
        st.session_state[f"{key}_uploader_index"] = str(random.randint(0, 100))
    if f"{key}_cached_files" not in st.session_state:
        st.session_state[f"{key}_cached_files"] = {}
    if f"{key}_cached_dfs" not in st.session_state:
        st.session_state[f"{key}_cached_dfs"] = {}
    
    files = st.file_uploader(
        upload_label,
        type=["csv"],
        accept_multiple_files=True,
        key=key + "_file_uploader_" + st.session_state[f"{key}_uploader_index"],
    )
    if files is not None:
        file_names = [file.name for file in files]
        for file in files:
            if file.name not in uploaded_files_var.value:
                current_files = list(uploaded_files_var.value)
                current_files.append(file.name)
                uploaded_files_var.value = current_files
            st.session_state[f"{key}_cached_files"][file.name] = file
    last_selected_file = st.session_state.get(f"{key}_last_selected_file", None)
    last_selected_df = st.session_state.get(f"{key}_last_selected_df", None)
    
    selected_file = st.selectbox(
        "Select a file to process",
        options=uploaded_files_var.value,
        key=f"{key}_file_select",
    )
    changed = selected_file != last_selected_file

    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        encoding = st.selectbox(
            "File encoding",
            disabled=len(uploaded_files_var.value) == 0,
            options=FILE_ENCODING_OPTIONS,
            key=f"{key}_encoding_db",
            index=FILE_ENCODING_OPTIONS.index(st.session_state[f"{key}_encoding"]),
        )
    with col2:
        st.number_input(
            "Maximum rows to process (0 = all)",
            disabled=len(uploaded_files_var.value) == 0,
            min_value=0,
            step=1000,
            key=max_rows_var.key,
        )
    with col3:
        st.text("")
        st.text("")
        reload = st.button("Reload", key=f"{key}_reload", disabled=len(uploaded_files_var.value) == 0)

    selected_df = pd.DataFrame()
    cache_key = f"{selected_file}_{encoding}_{max_rows_var.value}"
    
    if selected_file not in [None, ""] and (changed or reload or cache_key not in st.session_state[f"{key}_cached_dfs"]):
        st.session_state[f"{key}_encoding"] = encoding
        file_to_read = None
        
        if files is not None:
            for file in files:
                if file.name == selected_file:
                    file_to_read = file
                    break
        
        if file_to_read is None and selected_file in st.session_state[f"{key}_cached_files"]:
            file_to_read = st.session_state[f"{key}_cached_files"][selected_file]
        
        if file_to_read is not None:
            try:
                file_to_read.seek(0)
            except (AttributeError, OSError):
                pass
            
            selected_df = (
                pd.read_csv(
                    file_to_read,
                    encoding=encoding,
                    nrows=max_rows_var.value,
                    encoding_errors="ignore",
                    low_memory=False,
                )
                if max_rows_var.value > 0
                else pd.read_csv(
                    file_to_read,
                    encoding=encoding,
                    encoding_errors="ignore",
                    low_memory=False,
                )
            )
            selected_df.columns = [
                clean_for_column_name(col) for col in selected_df.columns
            ]
            st.session_state[f"{key}_cached_dfs"][cache_key] = selected_df
    elif selected_file not in [None, ""] and cache_key in st.session_state[f"{key}_cached_dfs"]:
        selected_df = st.session_state[f"{key}_cached_dfs"][cache_key]
    elif (selected_file in [None, ""] or len(uploaded_files_var.value) == 0) and last_selected_df is not None and len(last_selected_df) > 0:
        selected_df = last_selected_df
    
    if selected_df is not None and len(selected_df) > 0:
        st.dataframe(
            selected_df[:show_rows],
            hide_index=True,
            use_container_width=True,
            height=height,
        )
    st.session_state[f"{key}_last_selected_file"] = selected_file
    st.session_state[f"{key}_last_selected_df"] = selected_df
    changed = changed or reload
    return selected_file, selected_df, changed


def prepare_input_df(
    workflow, input_df_var, processed_df_var
):
    if f"{workflow}_last_identifier" not in st.session_state:
        st.session_state[f"{workflow}_last_identifier"] = ""
    if f"{workflow}_last_attributes" not in st.session_state:
        st.session_state[f"{workflow}_last_attributes"] = []
    if f"{workflow}_last_suppress_zeros" not in st.session_state:
        st.session_state[f"{workflow}_last_suppress_zeros"] = True
    if f"{workflow}_selected_binned_cols" not in st.session_state:
        st.session_state[f"{workflow}_selected_binned_cols"] = []
    if f"{workflow}_selected_binned_size" not in st.session_state:
        st.session_state[f"{workflow}_selected_binned_size"] = "Year"
    if f"{workflow}_selected_num_bins" not in st.session_state:
        st.session_state[f"{workflow}_selected_num_bins"] = 5
    if f"{workflow}_selected_trim_percent" not in st.session_state:
        st.session_state[f"{workflow}_selected_trim_percent"] = 0.0
    if f"{workflow}_selected_compound_cols" not in st.session_state:
        st.session_state[f"{workflow}_selected_compound_cols"] = []
    if f"{workflow}_rename_map" not in st.session_state:
        st.session_state[f"{workflow}_rename_map"] = {}
    if (
        f"{workflow}_intermediate_dfs" not in st.session_state
        or len(st.session_state[f"{workflow}_intermediate_dfs"]) == 0
    ):
        st.session_state[f"{workflow}_intermediate_dfs"] = {
            'input': input_df_var.value.copy(deep=True)
        }

    reload = False
    df_sequence = ["input", "selected", "datetime_bin", "numeric_bin", "expanded", "suppress_count", "suppress_null"]

    def df_updated(df_name, reset):
        # ensure columns are propagated forward
        index = df_sequence.index(df_name)
        input_df = st.session_state[f"{workflow}_intermediate_dfs"]["input"]
        last_df = st.session_state[f"{workflow}_intermediate_dfs"][df_name]
        prior_df = st.session_state[f"{workflow}_intermediate_dfs"][df_sequence[index-1]] if index > 0 else input_df
        for col in last_df.columns:
            # add to all subsequent dataframes if not present
            for df in df_sequence[df_sequence.index(df_name) + 1:]:
                if df in st.session_state[f"{workflow}_intermediate_dfs"]:
                    if col in st.session_state[f"{workflow}_selected_compound_cols"]: # and df_name in ["expanded", "suppress_count", "suppress_null"]:
                        pass
                    else:
                        st.session_state[f"{workflow}_intermediate_dfs"][df][col] = last_df[
                            col
                        ]
        if reset:
            for df in df_sequence[index:]:
                st.session_state[f"{workflow}_intermediate_dfs"][df] = prior_df.copy(deep=True)
        else:
            # for all subsequent dataframes, remove columns that are not in the current dataframe
            for df in df_sequence[index+1:]:
                if df in st.session_state[f"{workflow}_intermediate_dfs"]:
                    for col in st.session_state[f"{workflow}_intermediate_dfs"][df].columns:
                        if col in input_df.columns and col not in last_df.columns:
                            st.session_state[f"{workflow}_intermediate_dfs"][df].drop(columns=[col], inplace=True)

    def prepare_stage(df_name):
        index = df_sequence.index(df_name)
        if index > 0:
            last_df_name = df_sequence[index-1]
            last_df = st.session_state[f"{workflow}_intermediate_dfs"][last_df_name]
            if df_name not in st.session_state[f"{workflow}_intermediate_dfs"]:
                st.session_state[f"{workflow}_intermediate_dfs"][df_name] = last_df.copy(deep=True)
            this_df = st.session_state[f"{workflow}_intermediate_dfs"][df_name]
            return last_df, this_df
        else:
            input_df = st.session_state[f"{workflow}_intermediate_dfs"]["input"]
            return input_df, input_df

    input_df = st.session_state[f"{workflow}_intermediate_dfs"]["input"]

    st.markdown("### Prepare input data",
                help="Perform these steps in sequence as needed to create the prepared dataset.")

    last_df, this_df = prepare_stage("selected")
    with st.expander("Select attribute columns to include", expanded=False):
        st.warning('Note that input data must be formatted such that each row represents a single, unique data subject.')
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Select all", use_container_width=True):
                for col in input_df.columns.to_numpy():
                    st.session_state[f"{workflow}_{col}"] = True
        with b2:
            if st.button("Deselect all", use_container_width=True):
                for col in input_df.columns.to_numpy():
                    st.session_state[f"{workflow}_{col}"] = False
        for col in input_df.columns.to_numpy():
            if f"{workflow}_{col}" not in st.session_state:
                st.session_state[f"{workflow}_{col}"] = False

            input = st.checkbox(
                col,
                key=f"{workflow}_{col}_input",
                value=st.session_state[f"{workflow}_{col}"],
            )
            st.session_state[f"{workflow}_{col}"] = input

    selected_cols = [
        col
        for col in input_df.columns.to_numpy()
        if st.session_state[f"{workflow}_{col}"] is True
    ]

    if selected_cols != st.session_state[f"{workflow}_last_attributes"] or len(selected_cols) == 0:
        reload = True
        this_df.drop(this_df.index, inplace=True) # empty the dataframe
        this_df.drop(this_df.columns, axis=1, inplace=True)
        for col in selected_cols:
            this_df[col] = last_df[col].replace(
                "nan", ""
            )
        st.session_state[f"{workflow}_last_attributes"] = selected_cols
        df_updated("selected", False)

    # print(f'selected df: {st.session_state[f"{workflow}_intermediate_dfs"]["selected"]}')

    last_df, this_df = prepare_stage("datetime_bin")
    with st.expander("Quantize datetime attributes", expanded=False):
        # quantize numeric columns into bins
        selected_columns = [
            col
            for col in st.session_state[f"{workflow}_selected_binned_cols"]
            if col in input_df.columns
        ]
        selected_date_cols = st.multiselect(
            "Select datetime attribute to quantize",
            default=selected_columns,
            options=selected_cols,
            help="Select the datetime columns you want to quantize. Quantizing datetime columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.",
        )

        bin_size_options = ["Year", "Half", "Quarter", "Month", "Day"]
        bin_size = st.radio(
            "Select bin size",
            index=bin_size_options.index(
                st.session_state[f"{workflow}_selected_binned_size"]
            ),
            options=bin_size_options,
            help="Select the bin size for the datetime columns you want to quantize. Quantizing datetime columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Quantize selected columns", key="quantize_date"):
                reload = True
                st.session_state[f"{workflow}_selected_binned_cols"] = selected_date_cols
                st.session_state[f"{workflow}_selected_binned_size"] = bin_size
                for col in selected_date_cols:
                    result = quantize_datetime(
                        this_df, col, bin_size
                    )
                    this_df[col] = result
                    this_df[col].replace(
                        "nan", ""
                    )
                df_updated("datetime_bin", False)
        with c2:
            if st.button("Reset", key="reset_date"):
                reload = True
                df_updated("datetime_bin", True)

    # print(f'datetime_bin_df: {st.session_state[f"{workflow}_intermediate_dfs"]["datetime_bin"]}')

    last_df, this_df = prepare_stage("numeric_bin")
    with st.expander("Quantize numeric attributes", expanded=False):
        # quantize numeric columns into bins
        numeric_cols = last_df.select_dtypes(include=["float64", "int64", "Int64"]).columns.to_list()

        selected_numeric_cols = st.multiselect(
            "Select numeric attributes to quantize",
            numeric_cols,
            help="Select the numeric columns you want to quantize. Quantizing numeric columns into bins makes it easier to synthesize data, but reduces the amount of information in the data. If you do not select any columns, no binning will be performed.",
        )
        num_bins = st.number_input(
            "Target bins",
            value=st.session_state[f"{workflow}_selected_num_bins"],
            help="Target number of bins to use for each column. If 0, no binning will be performed. Fewer bins makes it easier to synthesize data, but reduces the amount of information in the data. More bins makes it harder to synthesize data, but preserves more information in the data. Actual number of bins may vary from target based on preferred bin sizes.",
        )
        trim_percent = st.number_input(
            "Trim percent",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(st.session_state[f"{workflow}_selected_trim_percent"]),
            help="Percent of values to trim from the top and bottom of each column before binning. This helps to reduce the impact of outliers on the binning process. For example, if trim percent is 0.05, the top and bottom 5% of values will be trimmed from each column before binning. If 0, no trimming will be performed.",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Quantize selected columns", key="quantize_numeric"):
                reload = True
                if num_bins > 0:
                    for col in selected_numeric_cols:
                        qd = quantize_numeric(
                            last_df, col, num_bins, trim_percent
                        )
                        this_df[col] = qd

                st.session_state[f"{workflow}_selected_num_bins"] = num_bins
                st.session_state[f"{workflow}_selected_trim_percent"] = trim_percent
                df_updated("numeric_bin", False)
        with c2:
            if st.button("Reset", key="reset_quantize"):
                reload = True
                df_updated("numeric_bin", True)

    # print(f'numeric bin df: {st.session_state[f"{workflow}_intermediate_dfs"]["numeric_bin"]}')

    last_df, this_df = prepare_stage("expanded")
    with st.expander("Expand compound values", expanded=False):
        options = [
            x for x in last_df.columns.to_numpy()
        ]
        # columns = [
        #     x
        #     for x in options
        #     if x not in st.session_state[f"{workflow}_selected_compound_cols"]
        # ]
        selected_compound_cols = st.multiselect(
            "Select compound columns to expand",
            options,
            key=f"{workflow}_selected_compound_cols",
            help="Select the columns you want to expand into separate columns. If you do not select any columns, no expansion will be performed.",
        )
        col_delimiter = st.text_input(
            "Column delimiter",
            value=",",
            help="The character used to separate values in compound columns. If the delimiter is not present in a cell, the cell will be left unchanged. Any quotes around the entire list or individual values will be removed before processing, as will any enclosing square brackets.",
        )
        prefix_type = st.radio(
            "Expanded column prefix type",
            options=["Source column", "Custom"],
            index=0,
            help="Select the type of prefix to add to each new column created from the compound column. If 'Source column' is selected, the prefix will be the name of the source column. If 'Custom' is selected, you can enter a custom prefix.",
        )
        if prefix_type == "Source column":
            prefix = ""
        else:
            prefix = st.text_input(
                "Custom prefix",
                value="",
                help="Prefix to add to each new column created from the compound column. If no prefix is provided, no prefix will be added.",
            )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Expand selected columns", key="expand_compound"):
                reload = True
                to_add = (selected_compound_cols, col_delimiter)
                if (
                    col_delimiter != ""
                    and to_add not in st.session_state[f"{workflow}_selected_compound_cols"]
                ):
                    # st.session_state[f"{workflow}_selected_compound_cols"].append(to_add)
                    for col in st.session_state[
                        f"{workflow}_selected_compound_cols"
                    ]:

                        if prefix_type == "Source column":
                            prefix = col
                        def convert_to_list(x):
                            if type(x) != str:
                                return []
                            if x[0] == "[" and x[-1] == "]":
                                x = x[1:-1]
                            vals = [y.strip() for y in x.split(col_delimiter)]
                            vals = [y[1:-1] if y[0] == '"' and y[-1] == '"' else y for y in vals if len(y) > 1]
                            vals = [y[1:-1] if y[0] == '\'' and y[-1] == '\'' else y for y in vals if len(y) > 1]
                            if prefix != "":
                                vals = [f"{prefix}{y}" for y in vals]
                            return vals
                        # add each value as a separate column with a 1 if the value is present in the compound column and None otherwise
                        values = last_df[col].apply(convert_to_list)
                        unique_values = {v for vals in values for v in vals}
                        unique_values = [x for x in unique_values if x != ""]
                        for val in unique_values:
                            st.session_state[f"{workflow}_{val}"] = False
                            this_df[val] = values.apply(
                                lambda x: "1" if val in x and val != "nan" else ""
                            )
                        if col in this_df.columns:
                            this_df.drop(columns=[col], inplace=True)
                df_updated("expanded", False)
        with c2:
            if st.button("Reset", key="reset_expand"):
                reload = True
                df_updated("expanded", True)

    # print(f'expanded df: {st.session_state[f"{workflow}_intermediate_dfs"]["expanded"]}')

    last_df, this_df = prepare_stage("suppress_count")
    with st.expander("Suppress insignificant attribute values", expanded=False):
        
        if f"{workflow}_min_count" not in st.session_state:
            st.session_state[f"{workflow}_min_count"] = 0
        last_min = st.session_state[f"{workflow}_min_count"]
        min_value = st.number_input(
            "Minimum value count",
            key=f"{workflow}_min_count_input",
            value=st.session_state[f"{workflow}_min_count"],
            help="Minimum count of an attribute value to be included in the sensitive dataset. If 0, no filtering will be performed.",
        )
        if min_value != last_min:
            reload = True
            st.session_state[f"{workflow}_min_count"] = min_value
            for col in last_df.columns:
                value_counts = last_df[col].value_counts()
                # convert to dict with value as key and count as value
                value_counts = dict(
                    zip([str(x) for x in value_counts.index], value_counts.values, strict=False)
                )
                # remove any values that are less than the minimum count
                if last_df[col].dtype == "str":
                    this_df[col] = last_df[col].apply(
                        lambda x: ""
                        if str(x) in value_counts and value_counts[str(x)] < min_value
                        else str(x)
                    )
                elif last_df[col].dtype == "float64":
                    this_df[col] = last_df[col].apply(
                        lambda x: np.nan
                        if str(x) in value_counts and value_counts[str(x)] < min_value
                        else x
                    )
                elif last_df[col].dtype == "int64":
                    this_df[col] = last_df[col].apply(
                        lambda x: -sys.maxsize
                        if str(x) in value_counts and value_counts[str(x)] < min_value
                        else x
                    )
                    this_df[col] = last_df[col].astype("Int64")
                    this_df[col] = last_df[col].replace(-sys.maxsize, np.nan)
                else: # catches object and other types
                    this_df[col] = last_df[col].apply(
                        lambda x: ""
                        if str(x) in value_counts and value_counts[str(x)] < min_value
                        else str(x)
                    )
            df_updated("suppress_count", False)
        # print(f'suppress count: {st.session_state[f"{workflow}_intermediate_dfs"]["suppress_count"]}')

        last_suppress_zeros = st.session_state[f"{workflow}_last_suppress_zeros"]
        suppress_zeros = st.checkbox(
            "Suppress boolean False / binary 0",
            key=f"{workflow}_suppress_zeros_input",
            value=True,
            help="For boolean columns, maps the value False to None. For binary columns, maps the number 0 to None. This is useful when only the presence of an attribute is important, not the absence."
        )

        last_df, this_df = prepare_stage("suppress_null")
        if suppress_zeros or suppress_zeros != last_suppress_zeros:
            st.session_state[f"{workflow}_last_suppress_zeros"] = suppress_zeros
            if suppress_zeros:
                this_df = df_functions.suppress_boolean_binary(last_df, this_df)
            else:
                this_df = last_df.copy(deep=True)
            if suppress_zeros != last_suppress_zeros:
                reload = True
            df_updated("suppress_null", False)
            
    
        # print(f'suppress null: {st.session_state[f"{workflow}_intermediate_dfs"]["suppress_null"]}')
    processed_df = this_df.copy(deep=True)
    processed_df.replace({"<NA>": np.nan}, inplace=True)
    processed_df.replace({"nan": ""}, inplace=True)
    processed_df.replace({"1.0": "1"}, inplace=True)
    with st.expander("Rename attributes", expanded=False):
        if len(processed_df) == 0:
            st.warning("Please select attributes to include in the prepared dataset.")
        else:
            renamed = False
            for col in this_df.columns:
                new_name = st.text_input(
                    f"Rename {col}",
                    key=f"{workflow}_rename_{col}",
                    value=col,
                    help="Rename the attribute to a more descriptive name.",
                )
                if col not in st.session_state[f"{workflow}_rename_map"].keys() or st.session_state[f"{workflow}_rename_map"][col] != new_name:
                    # print(f'renaming {col} to {new_name}')
                    st.session_state[f"{workflow}_rename_map"][col] = new_name
                    renamed = True
            if renamed:
                reload = True

    processed_df_var.value = processed_df
    for col, rename in st.session_state[f"{workflow}_rename_map"].items():
        processed_df_var.value.rename(columns={col: rename}, inplace=True)
        st.session_state[f"{workflow}_{rename}"] = st.session_state[f"{workflow}_{col}"]
    if reload and len(input_df) > 0 and len(processed_df) > 0:
        st.rerun()


def validate_ai_report(messages, result, show_status=True):
    if show_status:
        st.status(
            "Validating AI report and generating faithfulness score...",
            expanded=False,
            state="running",
        )
    messages_to_llm = utils.prepare_validation(messages, result)
    ai_configuration = UIOpenAIConfiguration().get_configuration()
    validation = OpenAIClient(ai_configuration).generate_chat(messages_to_llm, False)
    return json.loads(re.sub(r"```json\n|\n```", "", validation)), messages_to_llm


def generate_text(messages, callbacks=None, **kwargs):
    if callbacks is None:
        callbacks = []
    ai_configuration = UIOpenAIConfiguration().get_configuration()
    return OpenAIClient(ai_configuration).generate_chat(messages, callbacks=callbacks, **kwargs)

def create_markdown_callback(placeholder, prefix=""):
    def on(text):
        placeholder.markdown(prefix + text, unsafe_allow_html=True)

    on_callback = LLMCallback()
    on_callback.on_llm_new_token = on
    return on_callback


def remove_connection_bar(fn):
    def on(_):
        fn(_)

    on_callback = LLMCallback()
    on_callback.on_llm_new_token = on
    return on_callback


def build_validation_ui(
    report_validation, attribute_report_validation_messages, report_data, file_name
):
    mode = os.environ.get("MODE", Mode.DEV.value)
    if report_validation != {}:
        validation_status = st.status(
            label=f"LLM faithfulness score: {report_validation['score']}/5",
            state="complete",
        )
        with validation_status:
            st.write(report_validation["explanation"])

            if mode == Mode.DEV.value:
                obj = json.dumps(
                    {
                        "message": attribute_report_validation_messages,
                        "result": report_validation,
                        "report": report_data,
                    },
                    indent=4,
                )
                st.download_button(
                    "Download faithfulness evaluation",
                    use_container_width=True,
                    data=str(obj),
                    file_name=f"{file_name}_{get_current_time()}_messages.json",
                    mime="text/json",
                )

def check_ai_configuration(enforce_structured_output=False):
    ai_configuration = UIOpenAIConfiguration().get_configuration()
    if ai_configuration.api_key == "" and ai_configuration.api_type == "Open AI":
        st.warning("Please set your OpenAI API key in the Settings page.")
    if ai_configuration.model == "":
        st.warning("Please set your OpenAI model in the Settings page.")

    list_enforce_structured_output = [
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4.1",
        "gpt-4.1-mini",
    ]
    if (
        enforce_structured_output
        and ai_configuration.model not in list_enforce_structured_output
    ):
        st.warning(
            "Your current OpenAI model does not support this workflow. Please use the Settings page to use `gpt-4.1-mini` or `gpt-4.1` as OpenAI Deployment Name."
        )


def format_report_group_options(group_dict, existing_groups) -> str:
    return " & ".join([f"{key}: {group_dict[key]}" for key in existing_groups])