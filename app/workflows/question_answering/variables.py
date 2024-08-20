# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import random
import streamlit as st
from app.util.session_variable import SessionVariable
import python.question_answering.prompts as prompts

class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.text_to_chunks = SessionVariable({}, prefix)
        self.text_to_vectors = SessionVariable({}, prefix)
        self.upload_key = SessionVariable(random.randint(1, 100), prefix)
        self.concept_graph = SessionVariable(None, prefix)
        self.community_to_concepts = SessionVariable({}, prefix)
        self.concept_to_community = SessionVariable({}, prefix)
        self.chunk_to_concepts = SessionVariable({}, prefix)
        self.concept_to_chunks = SessionVariable({}, prefix)
        self.previous_chunk = SessionVariable({}, prefix)
        self.next_chunk = SessionVariable({}, prefix)
        self.relevant_chunks = SessionVariable([], prefix)
        self.partial_answers = SessionVariable([], prefix)
        self.last_question = SessionVariable("", prefix)
        self.final_report = SessionVariable("", prefix)
        self.semantic_search_depth = SessionVariable(10, prefix)
        self.structural_search_steps = SessionVariable(1, prefix)
        self.relational_search_depth = SessionVariable(10, prefix)
        self.relevance_test_batch_size = SessionVariable(5, prefix)
        self.relevance_test_limit = SessionVariable(30, prefix)
        self.report_validation_messages = SessionVariable("", prefix)
        self.report_validation = SessionVariable({}, prefix)
        self.system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.chunk_progress = SessionVariable("", prefix)
        self.answer_progress = SessionVariable("", prefix)
        self.answer_update_batch_size = SessionVariable(5, prefix)

    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
