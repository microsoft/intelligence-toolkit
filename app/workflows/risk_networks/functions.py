# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
from util.openai_wrapper import UIOpenAIConfiguration
from util.session_variables import SessionVariables

import toolkit.risk_networks.config as config
from toolkit.AI.base_embedder import BaseEmbedder
from toolkit.AI.local_embedder import LocalEmbedder
from toolkit.AI.openai_embedder import OpenAIEmbedder

sv_home = SessionVariables("home")


def embedder() -> BaseEmbedder:
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        if sv_home.local_embeddings.value:
            return LocalEmbedder(
                db_name=config.cache_name,
                max_tokens=ai_configuration.max_tokens,
            )
        return OpenAIEmbedder(
            configuration=ai_configuration,
            db_name=config.cache_name,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Error creating connection: {e}")
        st.stop()
