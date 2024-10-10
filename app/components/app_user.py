# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

import streamlit as st
from streamlit_cognito_auth import CognitoAuthenticator

from app.javascript.scripts import get_auth_user
from app.util.enums import Mode
from app.util.session_variables import SessionVariables


class AppUser:
    sv = None

    def __init__(self, sv=None):
        if sv is not None:
            self.sv = sv
        else:
            self.sv = SessionVariables("home")
        self.login()

    def _get_info(self):
        return self.sv.username.value

    def _set_user(self, username):
        self.sv.username.value = username

    def view_get_info(self):
        if self.sv.username.value:
            st.sidebar.write(f"Logged in as {self.sv.username.value}")

    def _view_error_info(self, return_value):
        st.warning(
            f"Could not resolve authenticated user from identity provider: {return_value}."
        )
        st.stop()

    def login(self):
        if (
            "COGNITO_POOL_ID" in os.environ
            and "COGNITO_APP_CLIENT_ID" in os.environ
            and "COGNITO_APP_CLIENT_SECRET" in os.environ
        ):
            authenticated = False
            try:
                cognito_authenticator = CognitoAuthenticator(
                    pool_id=os.environ["COGNITO_POOL_ID"],
                    app_client_id=os.environ["COGNITO_APP_CLIENT_ID"],
                    app_client_secret=os.environ["COGNITO_APP_CLIENT_SECRET"],
                    use_cookies=True,
                )
                authenticated = cognito_authenticator.login()
            except Exception as e:
                print(e)
                self._view_error_info(e)

            if not authenticated:
                self._view_error_info("Not authenticated")
            else:
                username = cognito_authenticator.get_username()
                self._set_user(username)
                with st.sidebar:
                    st.button(
                        "Logout", "logout_btn", on_click=cognito_authenticator.logout
                    )
        return
        if self.sv.mode.value != Mode.CLOUD.value:
            return
        return_value = get_auth_user()
        username = None
        if return_value == 0:
            pass  # this is the result before the actual value is returned
        elif isinstance(return_value, list) and len(return_value) > 0:
            username = return_value[0]["user_id"]
            self._set_user(username)
        else:
            self._view_error_info(return_value)
