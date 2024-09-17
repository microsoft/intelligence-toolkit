# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import random

import pandas as pd
import streamlit as st

import toolkit.compare_case_groups.prompts as prompts
from app.util.session_variable import SessionVariable


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.case_groups_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.case_groups_binned_df = SessionVariable(pd.DataFrame(), prefix)
        self.case_groups_final_df = SessionVariable(pd.DataFrame(), prefix)
        self.case_groups_last_file_name = SessionVariable("", prefix)
        self.case_groups_model_df = SessionVariable(pd.DataFrame(), prefix)
        self.case_groups_filters = SessionVariable([], prefix)
        self.case_groups_groups = SessionVariable([], prefix)
        self.case_groups_selected_groups = SessionVariable([], prefix)
        self.case_groups_aggregates = SessionVariable([], prefix)
        self.case_groups_temporal = SessionVariable("", prefix)
        self.case_groups_description = SessionVariable("", prefix)
        self.case_groups_top_groups = SessionVariable(0, prefix)
        self.case_groups_report = SessionVariable("", prefix)
        self.case_groups_report_validation_messages = SessionVariable("", prefix)
        self.case_groups_report_validation = SessionVariable({}, prefix)
        self.case_groups_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.case_groups_subject_identifier = SessionVariable("", prefix)
        self.case_groups_upload_key = SessionVariable(random.randint(1, 100), prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
