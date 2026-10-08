# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
from app.util.secrets_handler import SecretsHandler

_OPENAI_KEY = "openai_secret"


def get_api_key() -> str:
    """Return the OpenAI API key from the app secrets store."""
    try:
        return SecretsHandler().get_secret(_OPENAI_KEY) or ""
    except Exception:  # noqa: BLE001
        return ""
