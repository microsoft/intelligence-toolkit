# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from util.session_variable import SessionVariable
from javascript.styles import add_styles
import components.app_user as au
import components.app_terminator as at
import components.app_openai as ao
import components.app_mode as am
import streamlit as st

def load_multipage_app(sv = None):
    #Load user if logged in
    user = au.app_user()
    user.view_get_info()

    #Terminate app (if needed for .exe)
    terminator = at.app_terminator()
    terminator.terminate_app_btn()

    #OpenAI key set
    app_openai = ao.app_openai()
    app_openai.api_info()

    #Protected mode
    app_mode = am.app_mode()
    app_mode.config()

    add_styles()

    if sv:
        reset_workflow_button = st.sidebar.button(":warning: Reset workflow", use_container_width=True, help='Clear all data on this workflow and start over. CAUTION: This action can\'t be undone.')
        if reset_workflow_button:
            sv.reset_workflow()
            st.rerun()

