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

#### Text Intelligence

- [Story Generation](/Story_Generation): Generates story narratives for domain-specific education and training.   
""")

if __name__ == '__main__':
    main()