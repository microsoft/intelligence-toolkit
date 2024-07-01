# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os
import time

import keyboard
import psutil
import streamlit as st
from util.enums import Mode
from util.session_variables import SessionVariables


class app_terminator:
    
    sv = None
    
    def __init__(self, sv = None):
        if "off_btn_disabled" not in st.session_state:
            st.session_state.off_btn_disabled = False
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables('home')

    def _on_click(self):
        def click():
            st.session_state.off_btn_disabled = not st.session_state.off_btn_disabled
        return click

    def terminate_app_btn(self):
        mode = os.environ.get("MODE", Mode.DEV.value)
        if mode == Mode.EXE.value:
            exit_app = st.sidebar.button("🔴 Terminate application", disabled=st.session_state.off_btn_disabled, on_click=self._on_click)
            if exit_app:
                st.text("Shutting down application...")
                time.sleep(2)
                pid = os.getpid()
                keyboard.press_and_release('ctrl+w')
                p = psutil.Process(pid)
                p.terminate()