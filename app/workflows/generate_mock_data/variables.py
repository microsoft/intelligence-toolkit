# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st

from app.util.session_variable import SessionVariable
from intelligence_toolkit.generate_mock_data.api import GenerateMockData
from intelligence_toolkit.generate_mock_data.schema_builder import (
    create_boilerplate_schema,
)


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.workflow_object = SessionVariable(GenerateMockData(), prefix)
        self.schema = SessionVariable(create_boilerplate_schema(), prefix)
        self.num_records_overall = SessionVariable(100, prefix)
        self.records_per_batch = SessionVariable(20, prefix)
        self.duplicate_records_per_batch = SessionVariable(0, prefix)
        self.related_records_per_batch = SessionVariable(0, prefix)
        self.primary_record_array = SessionVariable('', prefix)
        self.generated_dfs = SessionVariable({}, prefix)
        self.generated_objects = SessionVariable([], prefix)
        self.final_object = SessionVariable({}, prefix)
        self.generation_guidance = SessionVariable('', prefix)
        self.text_generation_guidance = SessionVariable('', prefix)
        self.record_arrays = SessionVariable([], prefix)
        self.loaded_filename = SessionVariable('', prefix)
        self.generated_texts = SessionVariable([], prefix)
        self.generated_text_df = SessionVariable(None, prefix)
        self.uploaded_synthesis_files = SessionVariable([], prefix)
        self.synthesis_max_rows_to_process = SessionVariable(0, prefix)
        self.text_synthesis_temperature = SessionVariable(0.5, prefix)
        self.record_synthesis_temperature = SessionVariable(0.5, prefix)
        self.input_texts = SessionVariable([], prefix)
        
    def reset_workflow(self):
        for key in st.session_state:
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
