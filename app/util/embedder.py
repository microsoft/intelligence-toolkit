from app.util.constants import LOCAL_EMBEDDING_MODEL_KEY
from app.util.openai_wrapper import UIOpenAIConfiguration
from app.util.secrets_handler import SecretsHandler
from intelligence_toolkit.AI.base_embedder import BaseEmbedder
from intelligence_toolkit.AI.local_embedder import LocalEmbedder
from intelligence_toolkit.AI.openai_embedder import OpenAIEmbedder
from intelligence_toolkit.query_text_data import config


def create_embedder(local_embedding: bool | None = False) -> BaseEmbedder:
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
        print(f"Error creating connection: {e}")