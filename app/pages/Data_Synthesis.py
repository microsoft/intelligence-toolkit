# noqa: N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
import app.workflows.data_synthesis.variables as ds_variables
import app.workflows.data_synthesis.workflow
from components.app_loader import load_multipage_app
from util.helper_fn import app_in_dev_mode

workflow = "data_synthesis"


def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="app/myapp.ico",
        page_title="Intelligence Toolkit | Data Synthesis",
    )
    sv = ds_variables.SessionVariables(workflow)
    load_multipage_app(sv)

    try:
        app.workflows.data_synthesis.workflow.create(sv, workflow)
    except Exception as e:
        if app_in_dev_mode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
