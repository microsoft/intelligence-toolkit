# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import workflows.risk_networks.workflow
from components.app_loader import load_multipage_app
import streamlit as st

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Risk Networks')
    load_multipage_app()
    workflows.risk_networks.workflow.create()

if __name__ == '__main__':
    main()