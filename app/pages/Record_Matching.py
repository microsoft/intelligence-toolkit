# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from components.app_loader import load_multipage_app
import workflows.question_answering.workflow
import streamlit as st

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Record Matching')
    load_multipage_app()
    workflows.record_matching.workflow.create()

if __name__ == '__main__':
    main()