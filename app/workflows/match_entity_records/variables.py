# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import random

import polars as pl
import streamlit as st

import toolkit.match_entity_records.prompts as prompts
from app.util.session_variable import SessionVariable
from toolkit.match_entity_records.config import DEFAULT_MAX_RECORD_DISTANCE


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.matching_uploaded_files = SessionVariable([], prefix)
        self.matching_dfs = SessionVariable({}, prefix)
        self.matching_merged_df = SessionVariable(pl.DataFrame(), prefix)
        self.matching_matches_df = SessionVariable(pl.DataFrame(), prefix)
        self.matching_max_rows_to_process = SessionVariable(0, prefix)
        self.matching_mapped_atts = SessionVariable([], prefix)
        self.matching_sentence_pair_scores = SessionVariable([], prefix)
        self.matching_sentence_pair_jaccard_threshold = SessionVariable(0.0, prefix)
        self.matching_sentence_pair_embedding_threshold = SessionVariable(
            DEFAULT_MAX_RECORD_DISTANCE, prefix
        )
        self.matching_last_sentence_pair_embedding_threshold = SessionVariable(
            0.001, prefix
        )
        self.matching_evaluations = SessionVariable(pl.DataFrame(), prefix)
        self.matching_report_validation = SessionVariable({}, prefix)
        self.matching_report_validation_messages = SessionVariable("", prefix)
        self.matching_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.matching_upload_key = SessionVariable(random.randint(1, 100), prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
