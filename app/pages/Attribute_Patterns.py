# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from components.app_loader import load_multipage_app
import workflows.attribute_patterns.workflow
import streamlit as st

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Attribute Patterns')
    load_multipage_app()
    workflows.attribute_patterns.workflow.create()

if __name__ == '__main__':
    main()