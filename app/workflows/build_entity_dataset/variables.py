# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import json

import streamlit as st
from util.session_variable import SessionVariable

from intelligence_toolkit.build_entity_dataset.api import BuildEntityDataset
from intelligence_toolkit.build_entity_dataset import config


class SessionVariables:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix
        self.create_session(prefix)

    def create_session(self, prefix):
        self.workflow_object = SessionVariable(BuildEntityDataset(), prefix)

        # Task definition
        self.bed_category = SessionVariable("", prefix)
        self.bed_guidance = SessionVariable("", prefix)
        self.bed_schema_json = SessionVariable("", prefix)
        self.bed_model = SessionVariable(config.DEFAULT_MODEL, prefix)
        self.bed_max_queries = SessionVariable(config.DEFAULT_MAX_QUERIES, prefix)
        self.bed_concurrency = SessionVariable(config.DEFAULT_CONCURRENCY, prefix)
        self.bed_budget = SessionVariable(config.DEFAULT_BUDGET, prefix)
        self.bed_verify = SessionVariable(False, prefix)

        # Export / theme config
        self.bed_title = SessionVariable("", prefix)
        self.bed_subtitle = SessionVariable("", prefix)
        self.bed_dataset_label = SessionVariable("entities", prefix)
        self.bed_primary_color = SessionVariable("#002B49", prefix)
        self.bed_accent_color = SessionVariable("#0078D4", prefix)
        self.bed_view_table = SessionVariable(True, prefix)
        self.bed_view_cards = SessionVariable(True, prefix)
        self.bed_view_network = SessionVariable(True, prefix)

        # Safety scan
        self.bed_safety_prompt = SessionVariable("", prefix)
        self.bed_safety_findings = SessionVariable([], prefix)
        self.bed_safety_dismissed = SessionVariable([], prefix)

    def reset_workflow(self):
        for key in list(st.session_state.keys()):
            if key.startswith(self.prefix):
                del st.session_state[key]
        self.create_session(self.prefix)
