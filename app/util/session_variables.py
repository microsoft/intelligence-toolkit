from util.enums import Mode
from util.session_variable import SessionVariable
import pandas as pd
import util.session_variable as sv
import os

class SessionVariables:

    def __init__(self, prefix = ''):
        self.narrative_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.mode = sv.SessionVariable(os.environ.get("MODE", Mode.DEV.value))
        self.username = sv.SessionVariable('')
        self.generation_model = sv.SessionVariable('gpt-4-turbo')
        self.embedding_model = sv.SessionVariable('text-embedding-ada-002')
