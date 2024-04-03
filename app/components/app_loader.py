# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from javascript.styles import add_styles
import components.app_user as au
import streamlit as st

def load_multipage_app():
    #Load user if logged in
    user = au.app_user()
    user.view_get_info()

    #load css
    # add_styles()

