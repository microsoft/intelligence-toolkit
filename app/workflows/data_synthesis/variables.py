# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import random

import pandas as pd
import streamlit as st
from util.session_variable import SessionVariable


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.synthesis_raw_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_processing_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_last_sensitive_file_name = SessionVariable("", prefix)
        self.synthesis_last_synthetic_file_name = SessionVariable("", prefix)
        self.synthesis_last_aggregate_file_name = SessionVariable("", prefix)
        self.synthesis_subject_identifier = SessionVariable("", prefix)
        self.synthesis_synthetic_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_aggregate_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_sen_agg_rep = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_sen_syn_rep = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_epsilon = SessionVariable(12.0, prefix)
        self.synthesis_delta = SessionVariable(0.0, prefix)
        self.synthesis_wide_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_min_count = SessionVariable(0, prefix)
        self.synthesis_suppress_zeros = SessionVariable(False, prefix)
        self.synthesis_last_suppress_zeros = SessionVariable(False, prefix)
        self.synthesis_upload_key = SessionVariable(random.randint(1, 100), prefix)
        self.synthesis_synthetic_upload_key = SessionVariable(
            random.randint(101, 200), prefix
        )
        self.synthesis_aggregate_upload_key = SessionVariable(
            random.randint(201, 300), prefix
        )

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
