# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from sklearn.kernel_approximation import svd
from util.session_variables import SessionVariables
import workflows.risk_networks.workflow
import workflows.risk_networks.variables as vars
from components.app_loader import load_multipage_app
import streamlit as st
from util.enums import Mode

workflow = 'risk_networks'
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Risk Networks')
    sv = vars.SessionVariables(workflow)
    
    load_multipage_app(sv)
    sv_home = SessionVariables('home')
    
    try:
        workflows.risk_networks.workflow.create(sv)
    except Exception as e:
        if sv_home.mode.value == Mode.DEV.value:
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()