# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import random

import pandas as pd
import streamlit as st
import workflows.question_answering.prompts as prompts
from util.session_variable import SessionVariable


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.answering_raw_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.answering_q_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.answering_next_file_id = SessionVariable(1, prefix)
        self.answering_next_chunk_id = SessionVariable(1, prefix)
        self.answering_next_q_id = SessionVariable(1, prefix)
        self.answering_files = SessionVariable({}, prefix)
        self.answering_surface_questions = SessionVariable({}, prefix)
        self.answering_deeper_questions = SessionVariable({}, prefix)
        self.answering_cluster_target = SessionVariable(10, prefix)
        self.answering_question_answers_df = SessionVariable(pd.DataFrame(), prefix)
        self.answering_question_network_df = SessionVariable(pd.DataFrame(), prefix)
        self.answering_report_text = SessionVariable("", prefix)
        self.answering_last_selections = SessionVariable(pd.DataFrame(), prefix)
        self.answering_last_question = SessionVariable("", prefix)
        self.answering_outline_limit = SessionVariable(4000, prefix)
        self.answering_answer_text = SessionVariable("", prefix)
        self.answering_last_lazy_question = SessionVariable("", prefix)
        self.answering_batch_size = SessionVariable(1, prefix)
        self.answering_lazy_outline = SessionVariable("", prefix)
        self.answering_lazy_answer_text = SessionVariable("", prefix)
        self.answering_outline = SessionVariable("", prefix)
        self.answering_max_tier = SessionVariable(2, prefix)
        self.answering_target_matches = SessionVariable(5, prefix)
        self.answering_status_history = SessionVariable("", prefix)
        self.answering_matches = SessionVariable("", prefix)
        self.answering_report_validation_messages = SessionVariable("", prefix)
        self.answering_report_validation = SessionVariable({}, prefix)
        self.answering_source_diversity = SessionVariable(1, prefix)
        self.answering_question_history = SessionVariable([], prefix)
        self.answering_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.answering_upload_key = SessionVariable(random.randint(1, 100), prefix)
        self.answering_max_iterations = SessionVariable(10, prefix)
        self.answering_context_list = SessionVariable([], prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
