# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from util.session_variables import SessionVariables
import workflows.attribute_patterns.workflow
import workflows.attribute_patterns.variables as vars
from components.app_loader import load_multipage_app
import streamlit as st
from util.enums import Mode

workflow = 'attribute_patterns'
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Attribute Patterns')
    sv = vars.SessionVariables(workflow)
    load_multipage_app(sv)
    
    sv_home = SessionVariables('home')

    try:
        workflows.attribute_patterns.workflow.create(sv)
    except Exception as e:
        if sv_home.mode.value == Mode.DEV.value:
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()