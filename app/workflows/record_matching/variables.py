# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from util.session_variable import SessionVariable
import polars as pl

import workflows.record_matching.prompts as prompts

class SessionVariables:

    def __init__(self, prefix):
        self.matching_uploaded_files = SessionVariable([], prefix)
        self.matching_dfs = SessionVariable({}, prefix)
        self.matching_merged_df = SessionVariable(pl.DataFrame(), prefix)
        self.matching_matches_df = SessionVariable(pl.DataFrame(), prefix)
        self.matching_max_rows_to_process = SessionVariable(0, prefix)
        self.matching_mapped_atts = SessionVariable([], prefix)
        self.matching_sentence_pair_scores = SessionVariable([], prefix)
        self.matching_sentence_pair_jaccard_threshold = SessionVariable(0.75, prefix)
        self.matching_sentence_pair_embedding_threshold = SessionVariable(0.05, prefix)
        self.matching_last_sentence_pair_embedding_threshold = SessionVariable(0.05, prefix)
        self.matching_evaluations = SessionVariable(pl.DataFrame(), prefix)
        self.matching_system_prompt = SessionVariable(prompts.list_prompts, prefix)
        self.matching_instructions = SessionVariable('', prefix)
        
