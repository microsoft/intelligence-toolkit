# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from streamlit_javascript import st_javascript
import streamlit as st

def get_auth_user():
    js_code = """await fetch("/.auth/me")
    .then(function(response) {return response.json();})
    """
    return st_javascript(js_code)