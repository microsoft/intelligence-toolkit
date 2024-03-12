import streamlit as st
import util.mermaid as mermaid

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title='Intelligence Toolkit | Home')

    st.markdown(f"""\
# Intelligence Toolkit

*Interactive workflows for creating AI intelligence reports from real-world data sources*

Project home: [github.com/microsoft/intelligence-toolkit](https://github.com/microsoft/intelligence-toolkit) 

##### Input Data Types

The Intelligence Toolkit supports workflows spanning two broad categories of data:
- **Case Data** describing individual people (e.g., users, respondants, patients, victims)
- **Entity Data** describing real-world entities (e.g., organizations, businesses, buyers, suppliers)
                
##### Case Intelligence Workflows

These workflows 

- **Data Synthesis**: Generates private datasets and data summaries from sensitive case records
- **Attribute Patterns**: Generates reports on attribute patterns detected in streams of case records
- **Group Narratives**: Generates reports by defining and comparing groups of case records

##### Entity Intelligence Workflows

- **Record Matching**: Generates evaluations of record matches detected across entity datasets
- **Network Analysis**: Generates reports on networks of entities sharing attributes and risks
- **Question Answering**: Generates answers to questions about entities in a document collection

#### Workflow Selection                

Use the diagram below to select an appropriate workflow, then open the workflow from the sidebar to begin
                       
"""
    )
    mermaid.mermaid(
        code = """\
flowchart TD
    Data[Input Data] --> IsStructured{Is Structured?} --> |Yes| StructuredData[Structured Records]
    IsStructured{Is Structured?} --> |No| UnstructuredData[Unstructured Texta]
    UnstructuredData[Unstructured Texts] --> TextSubjects{Text Subject?} --> |Entities| KnowledgeCorpus[Entity Knowledge Corpus]
    KnowledgeCorpus[Entity Knowledge Corpus] --> |Question Answering Workflow| V[AI Answer Reports]
    StructuredData[Structured Records] --> DataSubjects{Record Subject?} --> |Person| PersonalData[Case Records]
    DataSubjects{Record Subject?} --> |Entity| EntityData[Entity Records]
    PersonalData[Personal Case Records] --> |Data Synthesis Workflow| SyntheticData[Synthetic Case Records]
    SyntheticData[Synthetic Case Records] --> NonPersonalData[Non-Personal Records]
    EntityData[Entity Records] --> NonPersonalData[Non-Personal Records]
    NonPersonalData[Non-Personal Records] --> HasTime{Time Attributes?} --> |Yes| TimeBinnedData[Time-Binnable Case Records]
    TimeBinnedData[Time-Binnable Case Records] --> |Attribute Patterns Workflow| AttributePatterns[AI Pattern Reports]
    NonPersonalData[Non-Personal Records]--> HasGroups{Grouping Attributes?} --> |Yes| GroupedData[Attribute-Groupable Case Records]
    GroupedData[Attribute-Groupable Case Records] --> |Group Narratives Workflow| GroupNarratives[AI Group Reports]
    EntityData[Entity Records] --> HasInconsistencies{Inconsistent Attributes?} --> |Yes| UnlinkedEntityData[Text-Linkable Entity Records]
    UnlinkedEntityData[Text-Matchable Entity Records] --> |Record Matching Workflow| RecordLinking[AI Link Reports]
    EntityData[Entity Records] --> HasIdentifiers{Identifying Attributes?} --> |Yes| LinkedEntityData[Attribute-Linkable Entity Records]
    LinkedEntityData[Attribute-Linkable Entity Records] --> |Network Analysis Workflow| NetworkAnalysis[AI Network Reports]

    """, 
        height = 2000
    )

    
if __name__ == '__main__':
    main()
