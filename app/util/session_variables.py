from util.session_variable import SessionVariable
import pandas as pd
import util.session_variable as sv
import os

class SessionVariables:

    def __init__(self, prefix):
        self.narrative_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.mode = sv.SessionVariable(os.environ.get("MODE", "dev"))
        self.username = sv.SessionVariable('')


      
