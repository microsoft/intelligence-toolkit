# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import streamlit as st

import toolkit.match_entity_records.config as config
from app.util.constants import LOCAL_EMBEDDING_MODEL_KEY
from app.util.openai_wrapper import UIOpenAIConfiguration
from app.util.secrets_handler import SecretsHandler
from toolkit.AI.base_embedder import BaseEmbedder
from toolkit.AI.local_embedder import LocalEmbedder
from toolkit.AI.openai_embedder import OpenAIEmbedder


def embedder(local_embedding: bool | None = False) -> BaseEmbedder:
    try:
        ai_configuration = UIOpenAIConfiguration().get_configuration()
        secrets_handler = SecretsHandler()
        if local_embedding:
            return LocalEmbedder(
                db_name=config.cache_name,
                max_tokens=ai_configuration.max_tokens,
                model=secrets_handler.get_secret(LOCAL_EMBEDDING_MODEL_KEY) or None,
            )
        return OpenAIEmbedder(
            configuration=ai_configuration,
            db_name=config.cache_name,
        )
    except Exception as e:
        st.error(f"Error creating connection: {e}")
        st.stop()
