# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from openai import OpenAI
import streamlit as st
import os
from util.SecretsHandler import SecretsHandler
from openai import AzureOpenAI

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
                self._instance = get_openai_type(self._key)
            except Exception as e:
                raise Exception(f'OpenAI client not created: {e}')


        return self._instance
    
def get_key_env():
    return os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else '' 

def get_openai_type(key, endpoint = None, api_version = None):
    if 'OPENAI_TYPE' in os.environ and os.environ['OPENAI_TYPE'] == "AZURE":
        endpoint = os.environ['AZURE_OPENAI_ENDPOINT'] if 'AZURE_OPENAI_ENDPOINT' in os.environ else None
        api_version = os.environ['AZURE_OPENAI_VERSION'] if 'AZURE_OPENAI_VERSION' in os.environ else None
        return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)
    else:
        return OpenAI(api_key=key)