import streamlit as st
import my_session_variables as msv
import os

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Intelligence Toolkit | Home')
    sv = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        sv = msv.SessionVariables()
    else:
        sv = st.session_state['my_session_variables']

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    st.markdown(f"""\
# Intelligence Toolkit
- A suite of tools for generating intelligence from real-world data types, including entity records, case records, and text documents.
- Identifies and explains real-world patterns, relationships, and risks, while maintaining data provenance and data privacy.
- Combines generative AI with complementary data science capabilities in visual workflows that are accessible to domain experts.         

#### Case Intelligence

- [Record Extraction](/Record_Extraction): Extracts records of structured attributes from unstructured text notes.
- [Attribute Patterns](/Attribute_Patterns): Detects time-linked attribute patterns from streams of case records.
- [Data Synthesis](/Data_Synthesis): Generates private datasets and summaries from sensitive case records.

#### Entity Intelligence

- [Record Linking](/Record_Linking): Detects links between entity records based on the similarity of their attributes.
- [Activity Patterns](/Activity_Patterns): Detects links between entities based on the similarity of their activity.
- [Network Analysis](/Network_Analysis): Detects networks of closely-related entities based on shared attributes and/or activity.
                                        
#### Text Intelligence

- [Question Answering](/Question_Answering): Generates reports in response to user questions about a report collection.
- [Data Narratives](/Data_Narratives): Generates reports by combining and comparing information across datasets. 
- [Story Generation](/Story_Generation): Generates story narratives for domain-specific education and training.   
""")

if __name__ == '__main__':
    main()