# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from openai import OpenAI
import streamlit as st
import os
from util.SecretsHandler import SecretsHandler

key = 'openaikey'
class _OpenAI:
    _instance = None
    _key = None
    _secrets = None

    def __init__(self):
        self._secrets = SecretsHandler()

    def client(self):
        if self._secrets.get_secret(key) != '':
            api_key = self._secrets.get_secret(key)
        else:
            api_key = get_key_env()
        if api_key != self._key:
            self._key = api_key
            try:
                self._instance = OpenAI(api_key=self._key)
            except Exception as e:
                raise Exception(f'OpenAI client not created: {e}')


        return self._instance
    
def get_key_env():
    return os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else '' 