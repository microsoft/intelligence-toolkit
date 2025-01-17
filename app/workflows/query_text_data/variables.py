# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import streamlit as st

import intelligence_toolkit.query_text_data.prompts as prompts
from app.util.session_variable import SessionVariable
from intelligence_toolkit.query_text_data.api import QueryTextData


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.workflow_object = SessionVariable(QueryTextData(), prefix)
        self.answer_local_embedding_enabled = SessionVariable(False, prefix)
        self.query = SessionVariable("", prefix)
        self.anchored_query = SessionVariable("", prefix)
        self.final_report = SessionVariable("", prefix)
        self.target_chunks_per_cluster = SessionVariable(5, prefix)
        self.claim_search_depth = SessionVariable(10, prefix)
        self.search_type = SessionVariable("Source text", prefix)
        self.net_new_sources = SessionVariable(0, prefix)
        self.adjacent_test_steps = SessionVariable(1, prefix)
        self.community_relevance_tests = SessionVariable(10, prefix)
        self.relevance_test_batch_size = SessionVariable(5, prefix)
        self.relevance_test_budget = SessionVariable(20, prefix)
        self.irrelevant_community_restart = SessionVariable(5, prefix)
        self.report_validation_messages = SessionVariable("", prefix)
        self.report_validation = SessionVariable({}, prefix)
        self.system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.chunk_progress = SessionVariable("", prefix)
        self.answer_progress = SessionVariable("", prefix)
        self.show_search_process = SessionVariable(False, prefix)
        self.thematic_analysis = SessionVariable("", prefix)
        self.thematic_commentary = SessionVariable("", prefix)
        self.analysis_update_interval = SessionVariable(10, prefix)
        self.do_live_analysis = SessionVariable(True, prefix)
        self.do_live_commentary = SessionVariable(True, prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
