# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
from openai import OpenAI
import streamlit as st
import os
from util.SecretsHandler import SecretsHandler
from openai import AzureOpenAI

key = 'openaikey'
openai_type_key = 'openai_typekey'
openai_version_key = 'openai_versionkey'
openai_endpoint_key = 'openai_endpointkey'
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
                self._instance = self.get_openai_api()
            except Exception as e:
                raise Exception(f'OpenAI client not created: {e}')


        return self._instance
    
    def get_openai_type(self):
        environ = os.environ['OPENAI_TYPE'] if 'OPENAI_TYPE' in os.environ else None
        secret = self._secrets.get_secret(openai_type_key)
        return secret if len(secret) > 0 else environ

    def get_azure_openai_version(self):
        environ = os.environ['AZURE_OPENAI_VERSION'] if 'AZURE_OPENAI_VERSION' in os.environ else None
        secret = self._secrets.get_secret(openai_version_key)
        return secret if len(secret) > 0 else environ

    def get_azure_openai_endpoint(self):
        environ = os.environ['AZURE_OPENAI_ENDPOINT'] if 'AZURE_OPENAI_ENDPOINT' in os.environ else None
        secret = self._secrets.get_secret(openai_endpoint_key)
        return secret if len(secret) > 0 else environ

    def get_openai_api(self):
        if self.get_openai_type() == "Azure OpenAI":
            print('self._key', self._key)
            return AzureOpenAI(api_key=self._key, azure_endpoint=self.get_azure_openai_endpoint(), api_version=self.get_azure_openai_version())
        else:
            return OpenAI(api_key=self._key)
    
def get_key_env():
    return os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else '' 