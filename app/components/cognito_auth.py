import os
import streamlit as st

from streamlit_cognito_auth import CognitoAuthenticator

class AWSCognitoAuth:
    is_logged_in = False
    authenticator = None

    @staticmethod
    def is_cognito_configured():
        return "POOL_ID" in os.environ and "APP_CLIENT_ID" in os.environ and "APP_CLIENT_SECRET" in os.environ

    def __init__(self):
        if self.is_cognito_configured():

            pool_id = os.environ["POOL_ID"]
            app_client_id = os.environ["APP_CLIENT_ID"]
            app_client_secret = os.environ["APP_CLIENT_SECRET"]

            self.authenticator = CognitoAuthenticator(
                pool_id=pool_id,
                app_client_id=app_client_id,
                app_client_secret=app_client_secret,
                use_cookies=False
            )

    def logout(self):
        if self.is_cognito_configured():
            print("Logout")
            self.authenticator.logout()
