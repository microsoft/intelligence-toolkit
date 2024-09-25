# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st

from app.util.session_variables import SessionVariables


class AppMode:
    sv = None
    sv_network = None

    def __init__(self, sv=None):
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables("home")

    def config(self):
        cache = st.sidebar.toggle(
            "Save embeddings",
            value=self.sv.save_cache.value,
            help="Enable caching of embeddings to speed up the application.",
        )
        if cache != self.sv.save_cache.value:
            self.sv.save_cache.value = cache
            st.rerun()