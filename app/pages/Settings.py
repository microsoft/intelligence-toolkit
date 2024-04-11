import os
from components.app_loader import load_multipage_app
from util.openai_instance import get_key_env, key
from util.SecretsHandler import SecretsHandler
import streamlit as st
import time
from util.session_variables import SessionVariables

def on_change(handler, key = None, value = None):
    def change():
        handler.write_secret(key, value)
    return change

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Settings')
    load_multipage_app()
    st.header("Settings")
    sv = SessionVariables('home')

    secrets_handler = SecretsHandler()
    placeholder = "Enter key here..."
    secret = secrets_handler.get_secret(key)
    is_mode_cloud = sv.mode.value == 'cloud'

    secret_input = st.text_input('Enter your OpenAI key', type="password", disabled=is_mode_cloud, placeholder=placeholder, value=secret)
    
    if secret and len(secret) > 0:
        st.info("Your key is saved securely.")
        clear_btn = st.button("Clear local key")

        if clear_btn:
            on_change(secrets_handler, key, value='')()
            time.sleep(0.3)
            st.rerun()

    if secret_input and secret_input != secret:
        secrets_handler.write_secret(key, secret_input)
        st.rerun()
    elif get_key_env() == '':
        st.warning("No OpenAI key found in the environment. Please insert one above.")
    elif not secret_input and not secret: 
        st.info("Using key from the environment.")

if __name__ == "__main__":
    main()