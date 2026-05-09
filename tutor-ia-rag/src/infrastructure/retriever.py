"""Módulo de recuperación y generación para el pipeline RAG.

Responsabilidad exclusiva de infraestructura: búsqueda vectorial y llamada
al LLM. La lógica de negocio (umbral de similitud, respuestas de fallback)
pertenece a la capa de aplicación (use_cases.py).
"""
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import settings
from src.domain.entities import DocumentChunk
from src.domain.ports import IRetriever

logger = logging.getLogger(__name__)

_MAX_QUERY_LEN: int = 1000

_SYSTEM_PROMPT: str = (
    "INSTRUCCIÓN ABSOLUTA: Respondé en español usando ÚNICAMENTE "
    "la información del contexto provisto. "
    "PROHIBIDO usar conocimiento externo al contexto. "
    "Si la información no está en el contexto, respondé EXACTAMENTE con esta frase: "
    "'La información solicitada no está disponible en el material del curso.' "
    "No agregues texto adicional a esa frase."
)


def _sanitize_query(query: str) -> str:
    """Trunca y limpia la query para reducir superficie de prompt injection.

    Args:
        query: Texto original del usuario.

    Returns:
        Query truncada a ``_MAX_QUERY_LEN`` caracteres y sin espacios extremos.
    """
    return query.strip()[:_MAX_QUERY_LEN]


def _build_default_store() -> Chroma:
    """Construye el vector store con embeddings de Ollama (nomic-embed-text)."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        ),
        persist_directory=settings.chroma_persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _build_default_llm() -> ChatOllama:
    """Construye el LLM local con Ollama (llama3.2)."""
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


class QueryRetriever(IRetriever):
    """Recupera chunks relevantes y genera respuestas con Ollama (llama3.2).

    Implementa :class:`~src.domain.ports.IRetriever`. No contiene lógica
    de negocio. El umbral de similitud y las decisiones de negocio son
    responsabilidad de la capa de aplicación.

    Args:
        store: Vector store inyectado. Si es ``None`` se crea uno con Ollama.
        llm: LLM inyectado. Si es ``None`` se usa ChatOllama con llama3.2.
        k: Número máximo de chunks a recuperar. Por defecto ``5``.

    Example:
        >>> retriever = QueryRetriever()
        >>> results = retriever.retrieve("¿Qué es backpropagation?")
        >>> chunks = [chunk for chunk, _ in results]
        >>> answer = retriever.generate_answer("¿Qué es backpropagation?", chunks)
    """

    def __init__(
        self,
        store: Chroma | None = None,
        llm: BaseChatModel | None = None,
        k: int = 5,
    ) -> None:
        self._k = k
        self._store = store or _build_default_store()
        self._llm = llm or _build_default_llm()

    def retrieve(self, query: str) -> list[tuple[DocumentChunk, float]]:
        """Recupera los chunks más similares con sus scores de relevancia.

        Usa ``similarity_search_with_relevance_scores`` que normaliza los
        scores en [0.0, 1.0] con distancia coseno, donde 1.0 = idéntico.

        Args:
            query: Pregunta del usuario en lenguaje natural.

        Returns:
            Lista de tuplas ``(DocumentChunk, score)`` ordenadas por
            relevancia descendente. Score en [0.0, 1.0].

        Raises:
            RuntimeError: Si falla la búsqueda en ChromaDB.
        """
        sanitized = _sanitize_query(query)

        try:
            raw: list[tuple[Document, float]] = (
                self._store.similarity_search_with_relevance_scores(
                    sanitized, k=self._k
                )
            )
        except Exception as exc:
            logger.error("Error en búsqueda vectorial: %s", exc, exc_info=True)
            raise RuntimeError(f"Fallo en la búsqueda vectorial: {exc}") from exc

        return [
            (
                DocumentChunk(
                    id=str(i),
                    content=doc.page_content,
                    source=doc.metadata.get("source", "desconocido"),
                    metadata=doc.metadata,
                ),
                score,
            )
            for i, (doc, score) in enumerate(raw)
        ]

    def generate_answer(self, query: str, chunks: list[DocumentChunk]) -> str:
        """Genera una respuesta usando el LLM con el contexto recuperado.

        Usa roles explícitos (SystemMessage / HumanMessage) para reducir
        la superficie de prompt injection y mejorar la adherencia al contexto
        con modelos locales como llama3.2.

        Args:
            query: Pregunta original del usuario.
            chunks: Lista de DocumentChunk que conforman el contexto.

        Returns:
            Respuesta generada por el LLM como string.

        Raises:
            RuntimeError: Si falla la llamada al LLM.
        """
        context = "\n\n---\n\n".join(
            f"[Fuente: {chunk.source}]\n{chunk.content}" for chunk in chunks
        )
        sanitized_query = _sanitize_query(query)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Contexto:\n{context}\n\nPregunta: {sanitized_query}"
            ),
        ]

        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("Error al llamar al LLM: %s", exc, exc_info=True)
            raise RuntimeError(f"Fallo en la generación de respuesta: {exc}") from exc
