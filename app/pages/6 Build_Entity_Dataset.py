# noqa: N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import asyncio

import streamlit as st

import app.workflows.build_entity_dataset.variables as bed_variables
from app.components.app_loader import load_multipage_app
from app.util.helper_fn import app_in_dev_mode
from app.workflows.build_entity_dataset import workflow as bed_workflow

workflow = "build_entity_dataset"


async def main() -> None:
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="app/myapp.ico",
        page_title="Intelligence Toolkit | Build Entity Dataset",
    )
    sv = bed_variables.SessionVariables(workflow)

    load_multipage_app(sv)

    try:
        await bed_workflow.create(sv, workflow)
    except Exception as e:
        if app_in_dev_mode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
