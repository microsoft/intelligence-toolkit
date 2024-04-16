import os
from components.app_loader import load_multipage_app
from util.openai_instance import get_key_env, key, _OpenAI, openai_endpoint_key, openai_version_key, openai_type_key
from util.SecretsHandler import SecretsHandler
import streamlit as st
import time
from util.session_variables import SessionVariables
from util.enums import Mode

openai = _OpenAI()

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
    is_mode_cloud = sv.mode.value == Mode.CLOUD.value

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
    elif get_key_env() == '' and len(secret) == 0:
        st.warning("No OpenAI key found in the environment. Please insert one above.")
    elif not secret_input and not secret: 
        st.info("Using key from the environment.")
    
    st.divider()

    st.header("OpenAI Type")
    st.markdown("Select the OpenAI type you want to use.")
    types = ["OpenAI", "Azure OpenAI"]
    type_input = st.radio("OpenAI Type", types, index=types.index(openai.get_openai_type()) or 0)
    type = openai.get_openai_type()
    if type != type_input:
        on_change(secrets_handler, openai_type_key, type_input)()
        st.rerun()

    if type_input == "Azure OpenAI":
        col1, col2 = st.columns(2)
        with col1:
            endpoint = st.text_input("Azure OpenAI Endpoint", type="password", value=openai.get_azure_openai_endpoint())
            if endpoint != openai.get_azure_openai_endpoint():
                on_change(secrets_handler, openai_endpoint_key, endpoint)()
                st.rerun()
                
        with col2:
            version = st.text_input("Azure OpenAI Version", value=openai.get_azure_openai_version())
            if version != openai.get_azure_openai_version():
                on_change(secrets_handler, openai_version_key, version)()
                st.rerun()

if __name__ == "__main__":
    main()