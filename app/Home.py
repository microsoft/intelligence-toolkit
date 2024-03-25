import streamlit as st
import util.mermaid as mermaid
from streamlit_javascript import st_javascript
import util.session_variables

def get_user(sv):
    if sv.mode.value != 'cloud':
        return
    css='''
            [data-testid="stSidebarNavItems"] {
                max-height: 100vh
            }
        '''
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    js_code = """await fetch("/.auth/me")
    .then(function(response) {return response.json();})
    """
    return_value = st_javascript(js_code)

    username = None
    if return_value == 0:
        pass # this is the result before the actual value is returned 
    elif isinstance(return_value, list) and len(return_value) > 0:
        username = return_value[0]["user_id"]
        sv.username.value = username
        st.sidebar.write(f"Logged in as {username}")
    else:
        st.warning(f"Could not directly read username from azure active directory: {return_value}.")     

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title='Intelligence Toolkit | Home')
    sv = util.session_variables.SessionVariables('home')
    get_user(sv)

    transparency_faq = open('./app/TransparencyFAQ.md', 'r').read()
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
