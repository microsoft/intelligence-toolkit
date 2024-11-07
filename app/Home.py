# noqa N999
# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import inspect
import os
import os.path

import streamlit as st
import util.mermaid as mermaid
from components.app_loader import load_multipage_app

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
    parts_text_original = parts.split("## Getting Started")[0]
    parts = parts_text_original.split("```mermaid")
    mermaid = (
        parts[1]
        .split("### How was Intelligence Toolkit evaluated?")[0]
        .replace("```", "")
    )

    parts_text = parts_text_original.split("### What workflow should I use?")[0]
    parts_text += parts_text_original.split(
        "### How was Intelligence Toolkit evaluated?"
    )[1]

    openai_text = content.split("</div>")[1].split("**6. Setting up the AI model:**")[1]
    parts_text += "## Getting Started"
    parts_text += openai_text.split(
        "**Option 2: Using Intelligence Toolkit as a Python Package (via PyPI)**"
    )[0]

    return parts_text, mermaid


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

    st.markdown("### What workflow should I use?")
    mermaid.mermaid(
        code=mermaid_text,
        height=1000,
    )



if __name__ == "__main__":
    main()