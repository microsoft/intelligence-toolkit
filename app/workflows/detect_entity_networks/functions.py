# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st
from itk_core.base_embedder import BaseEmbedder
from itk_core.local_embedder import LocalEmbedder
from itk_core.openai_embedder import OpenAIEmbedder
from util.openai_wrapper import UIOpenAIConfiguration

import intelligence_toolkit.detect_entity_networks.config as config
from app.util.constants import LOCAL_EMBEDDING_MODEL_KEY
from app.util.secrets_handler import SecretsHandler


def embedder(local_embedding: bool | None = True) -> BaseEmbedder:
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        secrets_handler = SecretsHandler()
        if local_embedding:
            return LocalEmbedder(
                db_name=config.cache_name,
                max_tokens=ai_configuration.max_tokens,
                concurrent_coroutines=80,
                model=secrets_handler.get_secret(LOCAL_EMBEDDING_MODEL_KEY) or None,
            )
        return OpenAIEmbedder(
            configuration=ai_configuration,
            db_name=config.cache_name,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Error creating connection: {e}")
        st.stop()
