from langchain_openai import OpenAIEmbeddings

from src.config import settings


def build_embeddings() -> OpenAIEmbeddings:
    """Factory que construye el modelo de embeddings de OpenAI."""
    return OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.openai_api_key,
    )
