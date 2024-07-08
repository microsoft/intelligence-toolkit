import os
import time

import streamlit as st
from components.app_loader import load_multipage_app
from util.constants import EMBEDDINGS_FILE_NAME, MAX_SIZE_EMBEDDINGS_KEY
from util.enums import Mode
from util.openai_wrapper import (
    UIOpenAIConfiguration,
    key,
    openai_azure_auth_type,
    openai_azure_model_key,
    openai_endpoint_key,
    openai_type_key,
    openai_version_key,
)
from util.SecretsHandler import SecretsHandler


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
    openai_config = UIOpenAIConfiguration().get_configuration()
    st.header("Settings")
    secrets_handler = SecretsHandler()
    
    secret = secrets_handler.get_secret(key)
    is_mode_cloud = os.environ.get("MODE", Mode.DEV.value) == Mode.CLOUD.value

    st.subheader("OpenAI Type")
    st.markdown("Select the OpenAI type you want to use.")
    types = ["OpenAI", "Azure OpenAI"]
    index = types.index(openai_config.api_type) if openai_config.api_type in types else 0
    type_input = st.radio("OpenAI Type", types, index=index, disabled=is_mode_cloud)
    
    if openai_config.api_type != type_input:
        on_change(secrets_handler, openai_type_key, type_input)()
        st.rerun()

    if type_input == "Azure OpenAI":
        types_az = ["Managed Identity", "Azure Key"]
        index_az = types_az.index(openai_config.az_auth_type) if openai_config.az_auth_type in types_az else 0
        type_input_az = st.radio("Azure OpenAI Auth Type", types_az, index=index_az, disabled=is_mode_cloud)
        if type_input_az != openai_config.az_auth_type:
            print('type_input_az', type_input_az)
            print('openai_config.az_auth_type', openai_config.az_auth_type)
            on_change(secrets_handler, openai_azure_auth_type, type_input_az)()
            st.rerun()
        col1, col2, col3 = st.columns(3)
        with col1:
            endpoint = st.text_input("Azure OpenAI Endpoint", disabled=is_mode_cloud, type="password", value=openai_config.api_base)
            if endpoint != openai_config.api_base:
                on_change(secrets_handler, openai_endpoint_key, endpoint)()
                st.rerun()
                
        with col2:
            deployment_name = st.text_input("Azure OpenAI Deployment Name", disabled=is_mode_cloud, value=openai_config.model)
            if deployment_name != openai_config.model:
                on_change(secrets_handler, openai_azure_model_key, deployment_name)()
                st.rerun()

        with col3:
            version = st.text_input("Azure OpenAI Version", disabled=is_mode_cloud, value=openai_config.api_version)
            if version != openai_config.api_version:
                on_change(secrets_handler, openai_version_key, version)()
                st.rerun()
    if type_input == "OpenAI" or type_input_az != "Managed Identity":
        placeholder = "Enter key here..."
        secret_input = st.text_input('Enter your key', type="password", disabled=is_mode_cloud, placeholder=placeholder, value=secret)
        
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
        elif openai_config.api_key == '':
            st.warning("No OpenAI key found in the environment. Please insert one above.")
        elif not secret_input and not secret: 
            st.info("Using key from the environment.")
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