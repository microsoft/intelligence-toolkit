# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
import streamlit as st
from util.openai_instance import get_key_env
from util.SecretsHandler import SecretsHandler

class app_openai:
    def _is_api_key_configured(self):
        secrets = SecretsHandler()
        if secrets.get_secret("api_key") != '':
            return True
        elif get_key_env() != '':
            return True
        return False

    def api_info(self):
        if not self._is_api_key_configured():
            st.error("No OpenAI key found in the environment. Please add it in the settings.")