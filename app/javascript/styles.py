# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st

style_sidebar = '''
    [data-testid="stSidebarNavItems"] {
        max-height: 100vh
    }
'''

style_pdf = '''body {
    font-family: 'helvetica';
}
'''

style_iframes = '''
    iframe {
        display: none;
    }
'''

def add_styles():
    st.markdown(f'''<style>
        {style_sidebar}
        {style_iframes}
    </style>''', unsafe_allow_html=True)
