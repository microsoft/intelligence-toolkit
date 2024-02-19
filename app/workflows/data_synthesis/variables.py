from util.session_variable import SessionVariable
import pandas as pd

class SessionVariables:

    def __init__(self, prefix):
        self.synthesis_raw_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_process_columns = SessionVariable([], prefix)
        self.synthesis_binned_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_last_file_name = SessionVariable('', prefix)
        self.synthesis_subject_identifier = SessionVariable('', prefix)
        self.synthesis_sensitive_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_synthetic_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_aggregate_df = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_sen_agg_rep = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_sen_syn_rep = SessionVariable(pd.DataFrame(), prefix)
        self.synthesis_epsilon = SessionVariable(12.0, prefix)
        self.synthesis_delta = SessionVariable(0.0, prefix)

