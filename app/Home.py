# noqa N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import inspect
import os
import os.path

import streamlit as st

import app.util.mermaid as mermaid
from app.components.app_loader import load_multipage_app

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


def get_readme_and_mermaid():
    file_path = os.path.join(path, "README.md")
    if not os.path.exists(file_path):
        file_path = os.path.join(path, "..", "README.md")
    with open(file_path) as file:
        content = file.read()
    folders = [
        f.name for f in os.scandir(os.path.join(path, "workflows")) if f.is_dir()
    ]
    for f in folders:
        content = content.replace(
            os.path.join(path, "workflows", "README.md"),
            f'{"_".join(word.capitalize() for word in f.split("_"))}',
        )
    parts = content.split("</div>")
    parts = "# Intelligence Toolkit" + parts[1]
    parts = parts.split("```mermaid")
    return parts[0], parts[1].split("## Diving Deeper")[0].replace("```", "")


def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=f"{path}myapp.ico",
        page_title="Intelligence Toolkit | Home",
    )

    load_multipage_app()
    transparency_faq, mermaid_text = get_readme_and_mermaid()

    st.markdown(transparency_faq)

    mermaid.mermaid(
        code=mermaid_text,
        height=1000,
    )



if __name__ == "__main__":
    main()