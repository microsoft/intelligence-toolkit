# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import pandas as pd
import util.session_variable as sv
from util.session_variable import SessionVariable


class SessionVariables:
    def __init__(self, prefix=""):
        self.narrative_input_df = SessionVariable(pd.DataFrame(), prefix)
        self.username = sv.SessionVariable("")
        self.generation_model = sv.SessionVariable("gpt-4-turbo")
        self.embedding_model = sv.SessionVariable("text-embedding-ada-002")
        self.protected_mode = sv.SessionVariable(False)
        self.max_embedding_size = sv.SessionVariable(500)
        self.save_cache = sv.SessionVariable(True)
        self.local_embeddings = sv.SessionVariable(False)
