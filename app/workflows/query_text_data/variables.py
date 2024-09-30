# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import random

import streamlit as st

import toolkit.query_text_data.prompts as prompts
from app.util.session_variable import SessionVariable
from toolkit.query_text_data.input_processor import PeriodOption


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.file_to_chunks = SessionVariable({}, prefix)
        self.cid_to_text = SessionVariable({}, prefix)
        self.cid_to_explained_text = SessionVariable({}, prefix)
        self.text_to_cid = SessionVariable({}, prefix)
        self.cid_to_vector = SessionVariable({}, prefix)
        self.upload_key = SessionVariable(random.randint(1, 100), prefix)
        self.period_concept_graphs = SessionVariable(None, prefix)
        self.hierarchical_communities = SessionVariable({}, prefix)
        self.community_to_label = SessionVariable({}, prefix)
        self.cid_to_concepts = SessionVariable({}, prefix)
        self.concept_to_cids = SessionVariable({}, prefix)
        self.previous_cid = SessionVariable({}, prefix)
        self.next_cid = SessionVariable({}, prefix)
        self.relevant_cids = SessionVariable([], prefix)
        self.partial_answers = SessionVariable([], prefix)
        self.last_question = SessionVariable("", prefix)
        self.final_report = SessionVariable("", prefix)
        self.period_to_cids = SessionVariable({}, prefix)
        self.node_to_period_to_pos = SessionVariable({}, prefix)
        self.node_to_period_to_shift = SessionVariable({}, prefix)
        self.cid_to_converging_pairs = SessionVariable({}, prefix)
        self.node_period_counts = SessionVariable({}, prefix)
        self.edge_period_counts = SessionVariable({}, prefix)
        self.cid_to_summary = SessionVariable({}, prefix)
        self.analysis_window_size = SessionVariable("NONE", prefix)
        self.hierarchical_clusters = SessionVariable(None, prefix)

        self.adjacent_chunk_steps = SessionVariable(1, prefix)
        self.community_relevance_tests = SessionVariable(10, prefix)
        self.relevance_test_batch_size = SessionVariable(5, prefix)
        self.relevance_test_budget = SessionVariable(100, prefix)
        self.answer_update_batch_size = SessionVariable(10, prefix)
        self.irrelevant_community_restart = SessionVariable(3, prefix)

        self.report_validation_messages = SessionVariable("", prefix)
        self.report_validation = SessionVariable({}, prefix)
        self.system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.chunk_progress = SessionVariable("", prefix)
        self.answer_progress = SessionVariable("", prefix)

        self.answer_local_embedding_enabled = SessionVariable(False, prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
