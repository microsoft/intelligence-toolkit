# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import random

import pandas as pd
import streamlit as st
from app.util.session_variable import SessionVariable

import toolkit.attribute_patterns.prompts as prompts


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.attribute_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_last_file_name = SessionVariable("", prefix)
        self.attribute_dynamic_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_min_pattern_count = SessionVariable(100, prefix)
        self.attribute_max_pattern_length = SessionVariable(5, prefix)
        self.attribute_node_to_period_to_pos = SessionVariable({}, prefix)
        # self.attribute_period_embeddings = SessionVariable([], prefix)
        # self.attribute_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_record_counter = SessionVariable(None, prefix)
        self.attribute_close_pairs = SessionVariable(0, prefix)
        self.attribute_all_pairs = SessionVariable(0, prefix)
        self.attribute_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_subject_identifier = SessionVariable("", prefix)
        self.attribute_binned_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.attribute_final_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_report = SessionVariable("", prefix)
        self.attribute_report_validation_messages = SessionVariable("", prefix)
        self.attribute_report_validation = SessionVariable({}, prefix)
        self.attribute_time_col = SessionVariable("", prefix)
        self.attribute_selected_pattern = SessionVariable("", prefix)
        self.attribute_selected_pattern_period = SessionVariable("", prefix)
        self.attribute_selected_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_selected_pattern_att_counts = SessionVariable(
            pd.DataFrame(), prefix
        )
        self.attribute_table_index = SessionVariable(0, prefix)
        self.attribute_upload_key = SessionVariable(random.randint(1, 100), prefix)
        self.attribute_time_series_df = SessionVariable(pd.DataFrame(), prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
