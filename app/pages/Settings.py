import os
from util.Embedder import Embedder
from components.app_loader import load_multipage_app
from util.openai_instance import get_key_env, key, _OpenAI, openai_endpoint_key, openai_version_key, openai_type_key
from util.SecretsHandler import SecretsHandler
import streamlit as st
import time
from util.session_variables import SessionVariables
from util.enums import Mode
from util.constants import MAX_SIZE_EMBEDDINGS_KEY, EMBEDDINGS_FILE_NAME

openai = _OpenAI()
def on_change(handler, key = None, value = None):
    def change():
        handler.write_secret(key, value)
    return change

def delete_embeddings_pickle(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == EMBEDDINGS_FILE_NAME:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon="app/myapp.ico", page_title='Intelligence Toolkit | Settings')
    load_multipage_app()

    st.header("Settings")
    sv = SessionVariables('home')
    secrets_handler = SecretsHandler()
    
    st.subheader("OpenAI Key")
    placeholder = "Enter key here..."
    secret = secrets_handler.get_secret(key)
    is_mode_cloud = sv.mode.value == Mode.CLOUD.value

    secret_input = st.text_input('Enter your OpenAI key', type="password", disabled=is_mode_cloud, placeholder=placeholder, value=secret)
    
    if secret and len(secret) > 0:
        st.info("Your key is saved securely.")
        clear_btn = st.button("Clear local key")

        if clear_btn:
            secrets_handler.delete_secret(key)
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

    st.subheader("OpenAI Type")
    st.markdown("Select the OpenAI type you want to use.")
    types = ["OpenAI", "Azure OpenAI"]
    index = types.index(openai.get_openai_type()) if openai.get_openai_type() in types else 0
    type_input = st.radio("OpenAI Type", types, index=index, disabled=is_mode_cloud)
    type = openai.get_openai_type()
    if type != type_input:
        on_change(secrets_handler, openai_type_key, type_input)()
        st.rerun()

    if type_input == "Azure OpenAI":
        col1, col2 = st.columns(2)
        with col1:
            endpoint = st.text_input("Azure OpenAI Endpoint", disabled=is_mode_cloud, type="password", value=openai.get_azure_openai_endpoint())
            if endpoint != openai.get_azure_openai_endpoint():
                on_change(secrets_handler, openai_endpoint_key, endpoint)()
                st.rerun()
                
        with col2:
            version = st.text_input("Azure OpenAI Version", disabled=is_mode_cloud, value=openai.get_azure_openai_version())
            if version != openai.get_azure_openai_version():
                on_change(secrets_handler, openai_version_key, version)()
                st.rerun()

    st.divider()
    st.subheader("Embeddings")
    max_size = int(secrets_handler.get_secret(MAX_SIZE_EMBEDDINGS_KEY) or 0)

    c1, c2 = st.columns(2)
    with c1:
        max_size_input = st.number_input('Max embeddings to store (per workflow)', min_value=0, value=max_size, step=1, format="%d", key="max_size", help="Set the maximum number of embeddings to cache. If reached, old embeddings will be deleted. If 0, there's no limit.")
        if max_size_input != max_size:
            on_change(secrets_handler, MAX_SIZE_EMBEDDINGS_KEY, max_size_input)()
            st.rerun()
    with c2:
        st.text('')
        st.text('')
        clear = st.button("Clear all embeddings")
        if clear:
            delete_embeddings_pickle(os.environ.get("CACHE_DIR", "cache"))

if __name__ == "__main__":
    main()