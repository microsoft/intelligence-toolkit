# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st
from javascript.scripts import get_auth_user
from util.session_variables import SessionVariables

class app_user:
    
    sv = None
    
    def __init__(self, sv = None):
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables('home')
        self.login()

    def _get_info(self):
        return self.sv.username.value 
    
    def _set_user(self, username):
        self.sv.username.value = username

    def view_get_info(self):
        if self.sv.username.value:
            st.sidebar.write(f"Logged in as {self.sv.username.value}")

    def _view_error_info(self, return_value):
        st.warning(f"Could not directly read username from azure active directory: {return_value}.")     

    def login(self):
        return
        if self.sv.mode.value != 'cloud':
            return
        return_value = get_auth_user()
        username = None
        if return_value == 0:
            pass # this is the result before the actual value is returned 
        elif isinstance(return_value, list) and len(return_value) > 0:
            username = return_value[0]["user_id"]
            self._set_user(username)
        else:
            self._view_error_info(return_value)
