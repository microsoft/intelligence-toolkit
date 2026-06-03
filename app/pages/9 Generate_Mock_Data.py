# noqa: N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import asyncio

import streamlit as st
import pandas as pd
from components.app_loader import load_multipage_app
from util.helper_fn import app_in_dev_mode

import app.workflows.generate_mock_data.variables as gmd_variables
import app.workflows.generate_mock_data.workflow as gmd_workflow

workflow = "generate_mock_data"

async def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="app/myapp.ico",
        page_title="Intelligence Toolkit | Generate Mock Data",
    )
    sv = gmd_variables.SessionVariables(workflow)
    load_multipage_app(sv)

    try:
        await gmd_workflow.create(sv, workflow)
    except Exception as e:
        if app_in_dev_mode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
