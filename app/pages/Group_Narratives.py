# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from util.session_variables import SessionVariables
import workflows.group_narratives.workflow
import workflows.group_narratives.variables as vars
import streamlit as st
from components.app_loader import load_multipage_app
from util.enums import Mode


workflow = 'group_narratives'

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Group Narratives')
    sv = vars.SessionVariables(workflow)
    load_multipage_app(sv)
    
    sv_home = SessionVariables('home')
    try:
        workflows.group_narratives.workflow.create(sv, workflow)
    except Exception as e:
        if sv_home.mode.value == Mode.DEV.value:
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()