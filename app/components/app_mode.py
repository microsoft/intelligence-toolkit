# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
import workflows.risk_networks.variables as rn_variables
from util.session_variables import SessionVariables


class AppMode:
    sv = None
    sv_network = None

    def __init__(self, sv=None):
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables("home")

        self.sv_network = rn_variables.SessionVariables("risk_networks")

    def config(self):
        mode = st.sidebar.toggle(
            "Protected mode",
            value=self.sv.protected_mode.value,
            help="Prevent entity identification on screen. Changing this value will reset the whole workflow on Risk Networks.",
        )
        cache = st.sidebar.toggle(
            "Save embeddings",
            value=self.sv.save_cache.value,
            help="Enable caching of embeddings to speed up the application.",
        )

        if mode != self.sv.protected_mode.value:
            self.sv.protected_mode.value = mode
            self.sv_network.reset_workflow()
            st.rerun()

        if cache != self.sv.save_cache.value:
            self.sv.save_cache.value = mode
            st.rerun()
