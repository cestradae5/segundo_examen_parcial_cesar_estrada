from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"

    # ChromaDB
    chroma_persist_dir: str = "vectorstore"
    chroma_collection: str = "curso_ia"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    retriever_top_k: int = 5
    similarity_threshold: float = 0.4  # calibrar con nomic-embed-text

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
