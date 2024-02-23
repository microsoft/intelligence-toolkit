from util.session_variable import SessionVariable
import pandas as pd

import workflows.attribute_patterns.prompts as prompts

class SessionVariables:

    def __init__(self, prefix):
        self.attribute_max_rows_to_process = SessionVariable(0, prefix)
        self.attribute_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_last_file_name = SessionVariable('', prefix)
        self.attribute_dynamic_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_min_pattern_count = SessionVariable(100, prefix)
        self.attribute_max_pattern_length = SessionVariable(5, prefix)
        self.attribute_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_node_to_centroid = SessionVariable({}, prefix)
        self.attribute_period_embeddings = SessionVariable([], prefix)
        self.attribute_embedding_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_record_counter = SessionVariable(None, prefix)
        self.attribute_close_pairs = SessionVariable(0, prefix)
        self.attribute_all_pairs = SessionVariable(0, prefix)
        self.attribute_pattern_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_subject_identifier = SessionVariable('', prefix)
        self.attribute_binned_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_wide_df = SessionVariable(pd.DataFrame(), prefix)
        self.attribute_min_count = SessionVariable(0, prefix)
        self.attribute_suppress_zeros = SessionVariable(False, prefix)
        self.attribute_last_suppress_zeros = SessionVariable(False, prefix)
        self.attribute_instructions = SessionVariable('', prefix)
        self.attribute_system_prompt = SessionVariable(prompts.system_prompt, prefix)
        self.attribute_final_df = SessionVariable(pd.DataFrame(), prefix)


