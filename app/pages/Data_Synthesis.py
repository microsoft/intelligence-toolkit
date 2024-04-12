# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from util.session_variables import SessionVariables
import workflows.data_synthesis.workflow
from components.app_loader import load_multipage_app
import streamlit as st
from util.enums import Mode


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Data Synthesis')
    load_multipage_app()
    sv_home = SessionVariables('home')
    try:
        workflows.data_synthesis.workflow.create()
    except Exception as e:
        if sv_home.mode.value == Mode.DEV.value:
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()