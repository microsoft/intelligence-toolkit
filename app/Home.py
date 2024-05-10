# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st
import os
from components.app_loader import load_multipage_app
import util.mermaid as mermaid

def get_readme():
    file_path = os.path.join('README.md')
    with open(file_path, 'r') as file:
        content = file.read()
    folders = [f.name for f in os.scandir('app/workflows') if f.is_dir()]
    for f in folders:
        content = content.replace(f'/app/workflows/{f}/README.md',f'/{"_".join(word.capitalize() for word in f.split("_"))}')
    return content.split('## Diving Deeper')[0]

    
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Home')
    load_multipage_app()
    transparency_faq = get_readme()

    st.markdown(transparency_faq + '\n\n' + f"""\
#### Which Intelligence Toolkit workflow is right for me and my data?

Use the diagram to identify an appropriate workflow, then open the workflow from the sidebar to the left.
""")

    mermaid.mermaid(
        code = """\
flowchart TD

    PersonalData[\\Personal Case Records/] ----> |Data Synthesis Workflow| SyntheticData[/Synthetic Case Records\\]
    EntityData[\\Entity Records/] ---> HasTime{Time Attributes?}
    CaseRecords[\\ Case Records/] ---> HasTime{Time Attributes?}
    HasTime{Time Attributes?} --> |Attribute Patterns Workflow| AttributePatterns[/AI Pattern Reports\\]
    EntityData[\\Entity Records/] ---> HasGroups{Grouping Attributes?}
    CaseRecords[\\Case Records/] ---> HasGroups{Grouping Attributes?}
    HasGroups{Grouping Attributes?} --> |Group Narratives Workflow| GroupNarratives[/AI Group Reports\\]
    EntityData[\\Entity Records/] ---> HasInconsistencies{Inconsistent Attributes?} --> |Record Matching Workflow| RecordLinking[/AI Match Reports\\]
    EntityData[\\Entity Records/] ---> HasIdentifiers{Identifying Attributes?} --> |Network Analysis Workflow| NetworkAnalysis[/AI Network Reports\\]
    EntityDocs[\\Entity Documents/] ----> |Question Answering Workflow| AnswerReports[/AI Answer Reports\\]

    """, 
        height = 600
    )
    
if __name__ == '__main__':
    main()
