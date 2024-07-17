# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import components.app_mode as am
import components.app_terminator as at
import components.app_user as au
import streamlit as st
from javascript.styles import add_styles


def load_multipage_app(sv=None):
    # Load user if logged in
    user = au.AppUser()
    user.view_get_info()

    # Terminate app (if needed for .exe)
    terminator = at.AppTerminator()
    terminator.terminate_app_btn()

    # Protected mode
    app_mode = am.AppMode()
    app_mode.config()

    add_styles()

    if sv:
        reset_workflow_button = st.sidebar.button(
            ":warning: Reset workflow",
            use_container_width=True,
            help="Clear all data on this workflow and start over. CAUTION: This action can't be undone.",
        )
        if reset_workflow_button:
            sv.reset_workflow()
            st.rerun()
