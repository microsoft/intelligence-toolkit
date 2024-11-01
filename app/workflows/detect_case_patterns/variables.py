# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import random

import pandas as pd
import streamlit as st

import intelligence_toolkit.detect_case_patterns.prompts as prompts
from intelligence_toolkit.detect_case_patterns import DetectCasePatterns
from app.util.session_variable import SessionVariable


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.workflow_object = SessionVariable(DetectCasePatterns(), prefix)
        self.detect_case_patterns_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_last_file_name = SessionVariable("", prefix)


        
        self.detect_case_patterns_dynamic_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_min_pattern_count = SessionVariable(100, prefix)
        self.detect_case_patterns_max_pattern_length = SessionVariable(5, prefix)
        self.detect_case_patterns_node_to_period_to_pos = SessionVariable({}, prefix)
        # self.detect_case_patterns_period_embeddings = SessionVariable([], prefix)
        # self.detect_case_patterns_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_record_counter = SessionVariable(None, prefix)
        self.detect_case_patterns_close_pairs = SessionVariable(0, prefix)
        self.detect_case_patterns_all_pairs = SessionVariable(0, prefix)
        self.detect_case_patterns_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_subject_identifier = SessionVariable("", prefix)
        self.detect_case_patterns_binned_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_system_prompt = SessionVariable(
            prompts.list_prompts, prefix
        )
        self.detect_case_patterns_final_df = SessionVariable(pd.DataFrame(), prefix)
        self.detect_case_patterns_report = SessionVariable("", prefix)
        self.detect_case_patterns_report_validation_messages = SessionVariable(
            "", prefix
        )
        self.detect_case_patterns_report_validation = SessionVariable({}, prefix)
        self.detect_case_patterns_time_col = SessionVariable("", prefix)
        self.detect_case_patterns_selected_pattern = SessionVariable("", prefix)
        self.detect_case_patterns_selected_pattern_period = SessionVariable("", prefix)
        self.detect_case_patterns_selected_pattern_df = SessionVariable(
            pd.DataFrame(), prefix
        )
        self.detect_case_patterns_selected_pattern_att_counts = SessionVariable(
            pd.DataFrame(), prefix
        )
        self.detect_case_patterns_table_index = SessionVariable(0, prefix)
        self.detect_case_patterns_time_series_df = SessionVariable(
            pd.DataFrame(), prefix
        )

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
