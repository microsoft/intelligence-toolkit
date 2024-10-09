import os
import streamlit as st

from streamlit_cognito_auth import CognitoAuthenticator

def add_cognito_auth():
    if "POOL_ID" in os.environ and "APP_CLIENT_ID" in os.environ and "APP_CLIENT_SECRET" in os.environ:

        pool_id = os.environ["POOL_ID"]
        app_client_id = os.environ["APP_CLIENT_ID"]
        app_client_secret = os.environ["APP_CLIENT_SECRET"]

        authenticator = CognitoAuthenticator(
            pool_id=pool_id,
            app_client_id=app_client_id,
            app_client_secret=app_client_secret,
            use_cookies=False
        )

        is_logged_in = authenticator.login()
        if not is_logged_in:
            st.stop()


        def logout():
            print("Logout")
            authenticator.logout()


        with st.sidebar:
            st.text(f"Welcome,\n{authenticator.get_username()}")
            st.button("Logout", "logout_btn", on_click=logout)
