# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import random

import pandas as pd
import streamlit as st

from app.util.session_variable import SessionVariable


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.anonymize_raw_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_processing_df = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_last_sensitive_file_name = SessionVariable("", prefix)
        self.anonymize_last_synthetic_file_name = SessionVariable("", prefix)
        self.anonymize_last_aggregate_file_name = SessionVariable("", prefix)
        self.anonymize_subject_identifier = SessionVariable("", prefix)
        self.anonymize_synthetic_df = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_aggregate_df = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_sen_agg_rep = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_sen_syn_rep = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_epsilon = SessionVariable(12.0, prefix)
        self.anonymize_delta = SessionVariable(0.0, prefix)
        self.anonymize_wide_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.anonymize_min_count = SessionVariable(0, prefix)
        self.anonymize_suppress_zeros = SessionVariable(False, prefix)
        self.anonymize_last_suppress_zeros = SessionVariable(False, prefix)
        self.anonymize_upload_key = SessionVariable(random.randint(1, 100), prefix)
        self.anonymize_synthetic_upload_key = SessionVariable(
            random.randint(101, 200), prefix
        )
        self.anonymize_aggregate_upload_key = SessionVariable(
            random.randint(201, 300), prefix
        )

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
