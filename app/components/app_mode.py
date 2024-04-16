# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st

from util.session_variables import SessionVariables

class app_mode:
    sv = None
    
    def __init__(self, sv = None):
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables('home')

    def config(self):
        mode = st.sidebar.toggle("Protected mode", value=self.sv.protected_mode.value, help="Prevent entity identification on screen.")
        if mode != self.sv.protected_mode.value:
            self.sv.protected_mode.value = mode
            st.rerun()
        