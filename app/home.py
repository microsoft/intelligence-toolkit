import streamlit as st
import util.mermaid as mermaid

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title='Intelligence Toolkit | Home')
    st.markdown(f'# Intelligence Toolkit')
    st.markdown('**A suite of interactive workflows for creating AI intelligence reports from real-world data sources**. For more information, visit [github.com/microsoft/intelligence-toolkit](https://github.com/microsoft/intelligence-toolkit)')
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""\
##### Use of Case Data

- **Units** are structured records describing individual people
- **Examples** include users, respondants, patients, victims
- **Analysis** aims to inform *policy* while preserving *privacy*
                
##### Use of Entity Data
                    
- **Units** are records or documents describing real-world entities
- **Examples** include organizations, countries, products, suppliers
- **Analysis** aims to understand *risks* carried by *relationships*
                
##### Case Intelligence Workflows

- **Data Synthesis** generates private datasets and data summaries from sensitive case records
- **Attribute Patterns** generates reports on attribute patterns detected in streams of case records
- **Group Narratives** generates reports by defining and comparing groups of case records

##### Entity Intelligence Workflows

- **Record Matching** generates evaluations of record matches detected across entity datasets
- **Network Analysis** generates reports on networks of entities sharing attributes and risks
- **Question Answering** generates answers to questions about entities in a document collection
                     
"""
    )
    with c2:
        st.markdown(f"""\
**Workflow selection**. Use the diagram to identify an appropriate workflow, then open the workflow from the sidebar to the left.
"""
        )

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
            height = 2000
        )

    
if __name__ == '__main__':
    main()
