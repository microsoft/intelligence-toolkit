# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from openai import OpenAI
import streamlit as st
import os
from util.SecretsHandler import SecretsHandler

class _OpenAI:
    _instance = None
    _key = None
    _secrets = None

    def __init__(self):
        self._secrets = SecretsHandler()

    def client(self):
        if self._secrets.get_secret("api_key") != '':
            key = st.secrets["api_key"]
        else:
            key = get_key_env()
        if key != self._key:
            self._key = key
            try:
                self._instance = OpenAI(api_key=key)
            except Exception as e:
                print(f'Error creating OpenAI client: {e}')

        return self._instance
    
def get_key_env():
    return os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else '' 