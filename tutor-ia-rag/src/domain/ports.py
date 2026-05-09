from abc import ABC, abstractmethod

from src.domain.entities import DocumentChunk


class IIndexer(ABC):
    @abstractmethod
    def index(self, docs_path: str) -> int: ...


class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> list[tuple[DocumentChunk, float]]: ...

    @abstractmethod
    def generate_answer(self, query: str, chunks: list[DocumentChunk]) -> str: ...
