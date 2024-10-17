# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import toolkit.AI.utils as utils
from app.util.df_functions import get_current_time, quantize_datetime, quantize_numeric
from app.util.download_pdf import add_download_pdf
from app.util.enums import Mode
from app.util.openai_wrapper import UIOpenAIConfiguration
from toolkit.AI.classes import LLMCallback
from toolkit.AI.client import OpenAIClient
from toolkit.AI.defaults import DEFAULT_MAX_INPUT_TOKENS


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


def report_download_ui(report_var, name):
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


def generative_ai_component(system_prompt_var, variables):
    st.markdown("##### Generative AI instructions")
    with st.expander(
        "Edit AI system prompt used to generate output report", expanded=True
    ):
        instructions_text = st.text_area(
            "Prompt text", value=system_prompt_var.value["user_prompt"], height=200
        )
        reset_prompt = st.button("Reset to default")

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
            st.warning(message)
    return generate, messages, reset_prompt


def generative_batch_ai_component(
    system_prompt_var, variables, batch_name, batch_val, batch_size
):
    st.markdown("##### Generative AI instructions")
    with st.expander("Edit AI System Prompt", expanded=True):
        instructions_text = st.text_area(
            "Contents of System Prompt used to generate AI outputs.",
            value=system_prompt_var.value["user_prompt"],
            height=200,
        )
        reset_prompt = st.button("Reset to default")

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


file_options = ["unicode-escape", "utf-8", "utf-8-sig"]
file_encoding_default = "unicode-escape"


def single_csv_uploader(
    workflow,
    upload_label,
    last_uploaded_file_name_var,
    input_df_var,
    processed_df_var,
    key,
    show_rows=10000,
    height=250,
):
    file = st.file_uploader(
        upload_label, type=["csv"], accept_multiple_files=False, key=key+'_file_uploader'
    )
    if f"{key}_encoding" not in st.session_state:
        st.session_state[f"{key}_encoding"] = file_encoding_default

    with st.expander("File options"):
        encoding = st.selectbox(
            "File encoding",
            options=file_options,
            key=f"{key}_encoding_sb",
            index=file_options.index(st.session_state[f"{key}_encoding"]),
        )

        reload = st.button("Reload", key=f"{key}_reload")
    if file is not None and (file.name != last_uploaded_file_name_var.value or reload):
        st.session_state[f"{key}_encoding"] = encoding
        last_uploaded_file_name_var.value = file.name
        df = pd.read_csv(
            file, encoding=encoding, encoding_errors="ignore", low_memory=False
        )
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
    # dfo = st.radio('Select data table', options=options, index=0, horizontal=True, key=f'{workflow}_{upload_label}_data_table_select')
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
        st.session_state[f"{key}_encoding"] = file_encoding_default

    files = st.file_uploader(
        upload_label, type=["csv"], accept_multiple_files=True, key=key+'_file_uploader'
    )
    file_names = [file.name for file in files] if files is not None else []
    uploaded_files_var.value = [v for v in uploaded_files_var.value if v in file_names]
    if files is not None:
        for file in files:
            if file.name not in uploaded_files_var.value:
                uploaded_files_var.value.append(file.name)
    last_selected_file = st.session_state.get(f"{key}_last_selected_file", None)
    selected_file = st.selectbox(
        "Select a file to process",
        options=uploaded_files_var.value if files else [],
        key=f"{key}_file_select",
    )
    changed = selected_file != last_selected_file
        
    with st.expander("File options"):
        st.number_input(
            "Maximum rows to process (0 = all)",
            min_value=0,
            step=1000,
            key=max_rows_var.key
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            encoding = st.selectbox(
                "File encoding",
                options=file_options,
                key=f"{key}_encoding_db",
                index=file_options.index(st.session_state[f"{key}_encoding"]),
            )
        with c2:
            reload = st.button("Reload", key=f"{key}_reload")

    selected_df = pd.DataFrame()
    if selected_file not in [None, ""] or reload:
        st.session_state[f"{key}_encoding"] = encoding
        for file in files:
            if file.name == selected_file:
                selected_df = (
                    pd.read_csv(
                        file,
                        encoding=encoding,
                        nrows=max_rows_var.value,
                        encoding_errors="ignore",
                        low_memory=False,
                    )
                    if max_rows_var.value > 0
                    else pd.read_csv(
                        file,
                        encoding=encoding,
                        encoding_errors="ignore",
                        low_memory=False,
                    )
                )
                break
        st.dataframe(
            selected_df[:show_rows],
            hide_index=True,
            use_container_width=True,
            height=height,
        )
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

    if "input" in st.session_state[f"{workflow}_intermediate_dfs"] and st.session_state[f"{workflow}_intermediate_dfs"]["input"].shape != input_df_var.value.shape:
        del st.session_state[f"{workflow}_suppress_zeros"]

    reload = False
    df_sequence = ["input", "selected", "datetime_bin", "numeric_bin", "expanded", "suppress_count", "suppress_null"]

    def df_updated(df_name):
        # ensure columns are propagated forward
        input_df = st.session_state[f"{workflow}_intermediate_dfs"]["input"]
        last_df = st.session_state[f"{workflow}_intermediate_dfs"][df_name]
        for col in last_df.columns:
            # add to all subsequent dataframes if not present
            for df in df_sequence[df_sequence.index(df_name) + 1:]:
                st.session_state[f"{workflow}_intermediate_dfs"][df][col] = last_df[col]
        # for all subsequent dataframes, remove columns that are not in the current dataframe
        for df in df_sequence[df_sequence.index(df_name) + 1:]:
            if df in st.session_state[f"{workflow}_intermediate_dfs"]:
                for col in st.session_state[f"{workflow}_intermediate_dfs"][df].columns:
                    if col in input_df.columns and col not in last_df.columns:
                        # Don't remove expanded columns
                        # TODO: How to remove expanded columns when the original column is removed?
                        st.session_state[f"{workflow}_intermediate_dfs"][df].drop(columns=[col], inplace=True)

    def prepare_stage(df_name):
        last_df_name = df_sequence[df_sequence.index(df_name)-1]
        last_df = st.session_state[f"{workflow}_intermediate_dfs"][last_df_name]
        if df_name not in st.session_state[f"{workflow}_intermediate_dfs"]:
            st.session_state[f"{workflow}_intermediate_dfs"][df_name] = last_df.copy(deep=True)
        this_df = st.session_state[f"{workflow}_intermediate_dfs"][df_name]
        return last_df, this_df

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
        df_updated("selected")

    # print(f'selected df: {st.session_state[f"{workflow}_intermediate_dfs"]["selected"]}')

    last_df, this_df = prepare_stage("datetime_bin")
    with st.expander("Quantize datetime attributes", expanded=False):
        # quantize numeric columns into bins
        selected_date_cols = st.multiselect(
            "Select datetime attribute to quantize",
            default=st.session_state[f"{workflow}_selected_binned_cols"],
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
            df_updated("datetime_bin")

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
            df_updated("numeric_bin")

    # print(f'numeric bin df: {st.session_state[f"{workflow}_intermediate_dfs"]["numeric_bin"]}')

    last_df, this_df = prepare_stage("expanded")
    with st.expander("Expand compound values", expanded=False):
        options = [
            x for x in last_df.columns.to_numpy()
        ]
        columns = [
            x
            for x in options
            if x not in st.session_state[f"{workflow}_selected_compound_cols"]
        ]
        selected_compound_cols = st.multiselect(
            "Select compound columns to expand",
            columns,
            help="Select the columns you want to expand into separate columns. If you do not select any columns, no expansion will be performed.",
        )
        col_delimiter = st.text_input(
            "Column delimiter",
            value="",
            help="The character used to separate values in compound columns. If the delimiter is not present in a cell, the cell will be left unchanged.",
        )

        if st.button("Expand selected columns", key="expand_compound"):
            reload = True
            to_add = (selected_compound_cols, col_delimiter)
            if (
                col_delimiter != ""
                and to_add not in st.session_state[f"{workflow}_selected_compound_cols"]
            ):
                st.session_state[f"{workflow}_selected_compound_cols"].append(to_add)
                for cols, delim in st.session_state[
                    f"{workflow}_selected_compound_cols"
                ]:
                    for col in cols:
                        # add each value as a separate column with a 1 if the value is present in the compound column and None otherwise
                        values = last_df.apply(
                            lambda x: [y.strip() for y in x.split(delim)]
                            if type(x) == str
                            else []
                        )
                        unique_values = {v for vals in values for v in vals}
                        unique_values = [x for x in unique_values if x != ""]
                        for val in unique_values:
                            st.session_state[f"{workflow}_{val}"] = False
                            this_df[val] = values.apply(
                                lambda x: "1" if val in x and val != "nan" else ""
                            )
                        this_df.drop(columns=[col], inplace=True)
            df_updated("expanded")
    
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
                    print(f'col: {col} is float64')
                    this_df[col] = last_df[col].apply(
                        lambda x: np.nan
                        if str(x) in value_counts and value_counts[str(x)] < min_value
                        else x
                    )
                elif last_df[col].dtype == "int64":
                    print(f'col: {col} is int64')
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
            df_updated("suppress_count")
        # print(f'suppress count: {st.session_state[f"{workflow}_intermediate_dfs"]["suppress_count"]}')

    
        initialized = True
        if f"{workflow}_suppress_zeros" not in st.session_state:
            print('Not initialized')
            st.session_state[f"{workflow}_suppress_zeros"] = len(this_df) > 0
            initialized = False

        suppress_zeros = st.checkbox(
            "Suppress boolean False / binary 0",
            key=f"{workflow}_suppress_zeros_input",
            value=st.session_state[f"{workflow}_suppress_zeros"] if initialized else True,
            help="For boolean columns, maps the value False to None. For binary columns, maps the number 0 to None. This is useful when only the presence of an attribute is important, not the absence.",
        )
        
        last_df, this_df = prepare_stage("suppress_null")


        if not initialized or suppress_zeros:
            st.session_state[f"{workflow}_suppress_zeros"] = suppress_zeros
            for col in last_df.columns:
                unique_values = list([str(x) for x in last_df[col].unique()])
                is_three_with_none = len(unique_values) == 3 and last_df[col].isna().any()
                if len(unique_values) <= 2 or is_three_with_none:
                    if "0" in unique_values or "0.0" in unique_values:
                        this_df[col] = last_df[col].astype(str).replace("0", np.nan).replace("0.0", np.nan)
                    elif 'False' in unique_values:
                        this_df[col] = last_df[col].astype(str).replace('False', np.nan)
            df_updated("suppress_null")
        if not suppress_zeros:
            for col in this_df.columns:
                unique_values = this_df[col].unique()
                is_three_with_none = (
                    len(unique_values) == 3 and this_df[col].isna().any()
                )
                if len(unique_values) <= 2 or is_three_with_none:
                    this_df[col] = last_df[col]
            df_updated("suppress_null")
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

def check_ai_configuration():
    ai_configuration = UIOpenAIConfiguration().get_configuration()
    if ai_configuration.api_key == "":
        st.warning("Please set your OpenAI API key in the Settings page.")
    if ai_configuration.model == "":
        st.warning("Please set your OpenAI model in the Settings page.")
