# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st
import workflows.record_matching.variables as vars
import workflows.record_matching.workflow
from components.app_loader import load_multipage_app
from util.helper_fn import app_in_dev_mode

workflow = "record_matching"


def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="app/myapp.ico",
        page_title="Intelligence Toolkit | Record Matching",
    )
    sv = vars.SessionVariables(workflow)
    load_multipage_app(sv)

    try:
        workflows.record_matching.workflow.create(sv, workflow)
    except Exception as e:
        if app_in_dev_mode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
