# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import workflows.question_answering.workflow
from components.app_loader import load_multipage_app
import streamlit as st


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Question Answering')
    load_multipage_app()
    workflows.question_answering.workflow.create()

if __name__ == '__main__':
    main()