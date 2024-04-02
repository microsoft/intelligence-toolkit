# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import workflows.data_synthesis.workflow
from components.app_loader import load_multipage_app
import streamlit as st

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Data Synthesis')
    load_multipage_app()
    workflows.data_synthesis.workflow.create()

if __name__ == '__main__':
    main()