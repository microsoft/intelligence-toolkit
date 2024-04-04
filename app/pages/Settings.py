import os
from util.openai_instance import get_key_env
from util.SecretsHandler import SecretsHandler
import streamlit as st
import time
from util.session_variables import SessionVariables

key = 'openaikey'
def on_change(handler, key = None, value = None):
    def change():
        handler.write_secret('api_key', st.session_state[key] if key else value)
    return change

def main():
    st.header("Settings")
    sv = SessionVariables('home')

    if key not in st.session_state:
        st.session_state[key] = ''
        
    secrets_handler = SecretsHandler()
    placeholder = "Enter key here..."
    secret = secrets_handler.get_secret("api_key")

    is_mode_cloud = sv.mode.value == 'cloud'

    secret_input = st.text_input('Enter your OpenAI key', key=key, type="password", disabled=is_mode_cloud, placeholder=placeholder, value=secret, on_change=on_change(secrets_handler, key))
    
    if secret and len(secret):
        st.info("Your key is saved securely.")
        clear_btn = st.button("Clear local key")

        if clear_btn:
            on_change(secrets_handler, value='')()
            time.sleep(0.3)
            st.rerun()

    if secret_input and secret_input != secret:
        st.rerun()
    elif get_key_env() == '':
        st.warning("No OpenAI key found in the environment. Please insert one above.")
    elif not secret_input and not secret: 
        st.info("Using key from the environment.")

if __name__ == "__main__":
    main()