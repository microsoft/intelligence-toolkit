# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

import hmac
import os

import streamlit as st
import toml

credentials_string = os.getenv("USER_CREDENTIALS")
credentials_dict = {}


def check_password() -> bool:
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("User not known or password incorrect")
    st.stop()
    return False


def load_passwords() -> None:
    if credentials_string:
        user_pairs = credentials_string.split(";")
        for pair in user_pairs:
            if ":" in pair:
                user, password = pair.split(":")
                credentials_dict[user] = password

        secrets_content = {"passwords": credentials_dict}

        secrets_file_path = os.path.join(".streamlit", "secrets.toml")
        if not os.path.exists(secrets_file_path):
            with open(secrets_file_path, "w") as secrets_file:
                toml.dump(secrets_content, secrets_file)
        else:
            with open(secrets_file_path) as secrets_file:
                current_secrets = toml.load(secrets_file)
                if current_secrets != secrets_content:
                    with open(secrets_file_path, "w") as secrets_file:
                        toml.dump(secrets_content, secrets_file)
                        print("User auth info updated")
