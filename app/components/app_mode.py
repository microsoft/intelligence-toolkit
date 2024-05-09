# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st

from util.session_variables import SessionVariables
import workflows.risk_networks.variables as rn_vars

class app_mode:
    sv = None
    sv_network = None
    
    def __init__(self, sv = None):
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables('home')

        self.sv_network = rn_vars.SessionVariables('risk_networks')

    def config(self):
        mode = st.sidebar.toggle("Protected mode", value=self.sv.protected_mode.value, help="Prevent entity identification on screen. Changing this value will reset the whole workflow on Risk Networks.")

        if mode != self.sv.protected_mode.value:
            self.sv.protected_mode.value = mode
            self.sv_network.reset_workflow()
            st.rerun()
        