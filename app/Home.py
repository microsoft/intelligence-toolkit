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


def get_readme():
    file_path = os.path.join(path, "README.md")
    if not os.path.exists(file_path):
        file_path = os.path.join(path, "..\\README.md")
    with open(file_path) as file:
        content = file.read()
    folders = [f.name for f in os.scandir(f"{path}\\workflows") if f.is_dir()]
    for f in folders:
        content = content.replace(
            f"{path}\\workflows\\{f}\\README.md",
            f'{"_".join(word.capitalize() for word in f.split("_"))}',
        )
    return content.split("## Diving Deeper")[0]


def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=f"{path}myapp.ico",
        page_title="Intelligence Toolkit | Home",
    )
    load_multipage_app()
    transparency_faq = get_readme()

    st.markdown(
        transparency_faq
        + "\n\n"
        + """\
#### Which Intelligence Toolkit workflow is right for my data and task?

Use the diagram to identify an appropriate workflow, then open the workflow from the sidebar to the left.
"""
    )

    mermaid.mermaid(
        code="""\
%%{init: {"flowchart": {"htmlLabels": true}} }%%
flowchart TD
    NoData["<b>Input</b>: None"] --> |"<b>Generate Record Data</b><br/>workflow"| MockData["Mock Data"]
    MockData["AI Mock Data"] --> PersonalData["<b>Input</b>: Personal Case Records"]
    MockData["AI Mock Data"] --> CaseRecords["<b>Input</b>: Case Records"]
    MockData["AI Mock Data"] --> EntityData["<b>Input</b>: Entity Records"]
    PersonalData["<b>Input</b>: Personal Case Records"] ----> |"<b>Anonymize Case Data</b><br/>workflow"| AnonData["Anonymous Case Records"]
    EntityData["<b>Input</b>: Entity Records"] ---> HasTime{"Time<br/>Attributes?"}
    CaseRecords["<b>Input</b>: Case Records"] ---> HasTime{"Time<br/>Attributes?"}
    HasTime{"Time<br/>Attributes?"} --> |"<b>Detect Case Patterns</b><br/>workflow"| CasePatterns["AI Pattern Reports"]
    EntityData["<b>Input</b>: Entity Records"] ---> HasGroups{"Grouping<br/>Attributes?"}
    CaseRecords["<b>Input</b>: Case Records"] ---> HasGroups{"Grouping<br/>Attributes?"}
    HasGroups{"Grouping<br/>Attributes?"} --> |"<b>Compare Case Records</b><br/>workflow"| MatchedEntities["AI Group Reports"]
    EntityData["<b>Input</b>: Entity Records"] ---> HasInconsistencies{"Inconsistent<br/>Attributes?"} --> |"<b>Match Entity Records</b><br/>workflow"| RecordLinking["AI Match Reports"]
    EntityData["<b>Input</b>: Entity Records"] ---> HasIdentifiers{"Identifying<br/>Attributes?"} --> |"<b>Detect Entity Networks</b><br/>workflow"| NetworkAnalysis["AI Network Reports"]
    TextDocs["<b>Input:</b> Text Data"] ------> |"<b>Query Text Data</b><br/>workflow"| AnswerReports["AI Answer Reports"]

    """,
        height=1000,
    )


if __name__ == "__main__":
    main()
