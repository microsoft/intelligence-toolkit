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
        local_embed = st.sidebar.toggle(
            "Use local embeddings",
            value=self.sv.local_embeddings.value,
            help="Don't call OpenAI to embed, use a local library.",
        )

        if cache != self.sv.save_cache.value:
            self.sv.save_cache.value = cache
            st.rerun()

        if local_embed != self.sv.local_embeddings.value:
            self.sv.local_embeddings.value = local_embed
            st.rerun()
