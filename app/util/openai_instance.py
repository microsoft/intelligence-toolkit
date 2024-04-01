from openai import OpenAI
import streamlit as st
import os


class _OpenAI:
    _instance = None
    _key = None

    def client(cls):
        if "api_key" in st.secrets and st.secrets["api_key"] != '':
            key = st.secrets["api_key"]
        else:
            key = os.environ['OPENAI_API_KEY']
        if key != cls._key:
            cls._key = key
            try:
                cls._instance = OpenAI(api_key=key)
            except Exception as e:
                print(f'Error creating OpenAI client: {e}')

        return cls._instance