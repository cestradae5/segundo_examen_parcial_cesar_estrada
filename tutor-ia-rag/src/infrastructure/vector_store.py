from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import settings
from src.domain.ports import IIndexer
from src.infrastructure.document_loader import LangChainDocumentLoader
from src.infrastructure.embedder import build_embeddings
from src.infrastructure.text_splitter import ChunkSplitter


class ChromaVectorStore(IIndexer):
    """Adapter de ChromaDB usando LangChain. Implementa IIndexer."""

    def __init__(self):
        self._store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=build_embeddings(),
            persist_directory=settings.chroma_persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def index(self, docs_path: str) -> int:
        loader = LangChainDocumentLoader()
        splitter = ChunkSplitter()

        docs = loader.load(docs_path)
        chunks = splitter.split(docs)

        if not chunks:
            return 0

        self._store.add_documents(chunks)
        return len(chunks)

    def as_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        top_k = k if k is not None else settings.retriever_top_k
        return self._store.as_retriever(search_kwargs={"k": top_k})

    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Recupera documentos con scores de relevancia normalizados en [0, 1].

        Args:
            query: Texto de búsqueda.
            k: Número de resultados. Usa ``settings.retriever_top_k`` si es None.

        Returns:
            Lista de tuplas ``(Document, score)`` ordenadas por relevancia.
        """
        top_k = k if k is not None else settings.retriever_top_k
        return self._store.similarity_search_with_relevance_scores(query, k=top_k)
