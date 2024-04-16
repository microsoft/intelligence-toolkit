import os
from components.app_loader import load_multipage_app
from util.openai_instance import get_key_env, key, _OpenAI
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

    # st.header("OpenAI Models")
    # st.markdown("Select the OpenAI models you want to use. This modification will be valid for this session only.")
    # with st.spinner("Fetching models..."):
    #     try:
    #         openai_models = openai.client().models.list()
    #         #Removes deprecated models:
    #         # Source: https://platform.openai.com/docs/models/gpt-3-5-turbo
    #         deprecated = [
    #             'gpt-3.5-turbo-16k',
    #             'gpt-3.5-turbo-0613',
    #             'gpt-3.5-turbo-16k-0613'
    #         ]
    #         openai_models = [x for x in openai_models if x.id not in deprecated]
    #         # order by id string, reversed so new ones are showed first
    #         openai_models = sorted(openai_models, key=lambda x: x.id, reverse=True)
    #     except Exception as e: 
    #         st.error(f"Invalid key. Please check your OpenAI key. {e}")
    #         return
    # gpt_list = [x.id for x in openai_models if 'gpt' in x.id]
    # print('gpt_list', gpt_list)
    # embeddings_list = [x.id for x in openai_models if 'embedding' in x.id]
    # index_model = gpt_list.index(sv.generation_model.value) if sv.generation_model.value in gpt_list else 0
    # model_change = st.selectbox("OpenAI generative model", gpt_list, index=index_model)
    # st.caption("Note that not all models will have the same token limit, so the information on the workflow screen might be inaccurate.")
    # if model_change != sv.generation_model.value:
    #     sv.generation_model.value = model_change
    #     st.rerun()

    # embedding_model = embeddings_list.index(sv.embedding_model.value) if sv.embedding_model.value in embeddings_list else 0
    # embedding_model_change = st.selectbox("OpenAI embedding model", embeddings_list, index=embedding_model)
    # if embedding_model_change != sv.embedding_model.value:
    #     sv.embedding_model.value = embedding_model_change
    #     st.rerun()

if __name__ == "__main__":
    main()