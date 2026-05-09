from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings


class ChunkSplitter:
    """Divide documentos en chunks con overlap usando RecursiveCharacterTextSplitter."""

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)
