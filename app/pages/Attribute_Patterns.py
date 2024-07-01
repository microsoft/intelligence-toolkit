# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st
import workflows.attribute_patterns.variables as vars
import workflows.attribute_patterns.workflow
from components.app_loader import load_multipage_app
from util.helper_fn import appInDevMode

workflow = 'attribute_patterns'
def main() -> None:
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Attribute Patterns')
    sv = vars.SessionVariables(workflow)
    load_multipage_app(sv)
    
    try:
        workflows.attribute_patterns.workflow.create(sv, workflow)
    except Exception as e:
        if appInDevMode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()