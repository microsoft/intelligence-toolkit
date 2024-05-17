# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import components.app_mode as am
import components.app_terminator as at
import components.app_user as au
import streamlit as st
from javascript.styles import add_styles
from util.openai_wrapper import UIOpenAIConfiguration


def load_multipage_app(sv = None):
    #Load user if logged in
    openai_config = UIOpenAIConfiguration().get_configuration()
    user = au.app_user()
    user.view_get_info()

    #Terminate app (if needed for .exe)
    terminator = at.app_terminator()
    terminator.terminate_app_btn()

    #OpenAI key set
    if not openai_config.api_key:
        st.error("No OpenAI key found in the environment. Please add it in the settings.")

    #Protected mode
    app_mode = am.app_mode()
    app_mode.config()

    add_styles()

    if sv:
        reset_workflow_button = st.sidebar.button(":warning: Reset workflow", use_container_width=True, help='Clear all data on this workflow and start over. CAUTION: This action can\'t be undone.')
        if reset_workflow_button:
            sv.reset_workflow()
            st.rerun()

