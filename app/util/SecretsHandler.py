import os
import streamlit as st

class SecretsHandler:
    _instance = None
    _directory = ".streamlit"

    def __init__(self):
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
            with(open(os.path.join(self._directory, "secrets.toml"), "w")) as f:
                f.write("")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            
        return cls._instance

    def write_secret(self, key, value):
        with(open(os.path.join(self._directory, "secrets.toml"), "w")) as f:
            f.write(f"{key} = '{value}'")

    def get_secret(self, key) -> str:
        if st.secrets and key in st.secrets:
            return st.secrets[key]
        return ''
      