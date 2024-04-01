import os
from util.SecretsHandler import SecretsHandler
import streamlit as st
import util.session_variables
import time

key = 'openaikey'
def on_change(handler, key = None, value = None):
    def change():
        handler.write_secret('api_key', st.session_state[key] if key else value)
    return change

def main():
    sv = util.session_variables.SessionVariables('home')

    st.header("Settings")
    if key not in st.session_state:
        st.session_state[key] = ''
        
    secrets_handler = SecretsHandler()
    placeholder = "Enter key here..."
    secret = secrets_handler.get_secret("api_key")

    secret_input = st.text_input('Enter your OpenAI key', key=key, type="password", placeholder=placeholder, value=secret, on_change=on_change(secrets_handler, key))
    
    if secret and len(secret):
        st.info("Your key is saved securely.")
        clear_btn = st.button("Clear local key")

        if clear_btn:
            on_change(secrets_handler, value='')()
            time.sleep(0.3)
            st.rerun()

    if secret_input and secret_input != secret:
        st.rerun()
    elif not secret_input and not secret: 
        st.info("Using the key from the environment.")

if __name__ == "__main__":
    main()