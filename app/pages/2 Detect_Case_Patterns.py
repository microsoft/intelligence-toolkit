# noqa: N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
from components.app_loader import load_multipage_app

import app.workflows.detect_case_patterns.variables as ap_variables
import app.workflows.detect_case_patterns.workflow
from app.util.helper_fn import app_in_dev_mode

workflow = "detect_case_patterns"


def main() -> None:
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="app/myapp.ico",
        page_title="Intelligence Toolkit | Detect Case Patterns",
    )
    sv = ap_variables.SessionVariables(workflow)
    load_multipage_app(sv)

    try:
        app.workflows.detect_case_patterns.workflow.create(sv, workflow)
    except Exception as e:
        if app_in_dev_mode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
