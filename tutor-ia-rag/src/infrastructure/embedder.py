from langchain_ollama import OllamaEmbeddings

from src.config import settings


def build_embeddings() -> OllamaEmbeddings:
    """Factory que construye el modelo de embeddings con Ollama."""
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )
