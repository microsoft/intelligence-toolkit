# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
from util.openai_wrapper import UIOpenAIConfiguration
from util.session_variables import SessionVariables

import toolkit.risk_networks.config as config
from toolkit.AI.embedder import Embedder

sv_home = SessionVariables("home")


def embedder():
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        return Embedder(
            ai_configuration, config.cache_dir, sv_home.local_embeddings.value
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Error creating connection: {e}")
        st.stop()
