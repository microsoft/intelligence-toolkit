# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import workflows.group_narratives.workflow
import streamlit as st
from components.app_loader import load_multipage_app

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Group Narratives')
    load_multipage_app()
    workflows.group_narratives.workflow.create()

if __name__ == '__main__':
    main()