# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from util.session_variable import SessionVariable
import pandas as pd

import workflows.group_narratives.prompts as prompts

class SessionVariables:

    def __init__(self, prefix):
        self.narrative_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.narrative_binned_df = SessionVariable(pd.DataFrame(), prefix)
        self.narrative_final_df = SessionVariable(pd.DataFrame(), prefix)
        self.narrative_last_file_name = SessionVariable('', prefix)
        self.narrative_model_df = SessionVariable(pd.DataFrame(), prefix)
        self.narrative_filters = SessionVariable([], prefix)
        self.narrative_groups = SessionVariable([], prefix)
        self.narrative_selected_groups = SessionVariable([], prefix)
        self.narrative_aggregates = SessionVariable([], prefix)
        self.narrative_temporal = SessionVariable('', prefix)
        self.narrative_description = SessionVariable('', prefix)
        self.narrative_top_groups = SessionVariable(0, prefix)
        self.narrative_top_attributes = SessionVariable(0, prefix)
        self.narrative_instructions = SessionVariable('', prefix)
        self.narrative_report = SessionVariable('', prefix)
        self.narrative_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.narrative_subject_identifier = SessionVariable('', prefix)
