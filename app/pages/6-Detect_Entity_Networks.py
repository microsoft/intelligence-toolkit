# noqa: N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import asyncio

import streamlit as st

import app.workflows.detect_entity_networks.variables as rn_variables
import app.workflows.detect_entity_networks.workflow
from app.components.app_loader import load_multipage_app
from app.util.helper_fn import app_in_dev_mode

workflow = "detect_entity_networks"


async def main() -> None:
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="app/myapp.ico",
        page_title="Intelligence Toolkit | Detect Entity Networks",
    )
    sv = rn_variables.SessionVariables(workflow)

    load_multipage_app(sv)

    try:
        await app.workflows.detect_entity_networks.workflow.create(sv, workflow)
    except Exception as e:
        if app_in_dev_mode():
            st.exception(e)
        else:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
