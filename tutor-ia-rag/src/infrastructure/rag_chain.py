"""Adapter RAGChain que implementa IRetriever para uso desde la API (FastAPI).

Delega la búsqueda vectorial en ChromaVectorStore y la generación en ChatOllama.
La lógica de negocio (umbral de similitud, fallbacks) pertenece a AskTutorUC.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config import settings
from src.domain.entities import DocumentChunk
from src.domain.ports import IRetriever
from src.infrastructure.vector_store import ChromaVectorStore

_SYSTEM_PROMPT = (
    "INSTRUCCIÓN ABSOLUTA: Respondé en español usando ÚNICAMENTE "
    "la información del contexto provisto. "
    "PROHIBIDO usar conocimiento externo al contexto. "
    "Si la información no está en el contexto, respondé EXACTAMENTE con esta frase: "
    "'La información solicitada no está disponible en el material del curso.' "
    "No agregues texto adicional a esa frase."
)


class RAGChain(IRetriever):
    """Implementa IRetriever delegando en ChromaVectorStore y ChatOllama.

    Args:
        vector_store: Instancia de ChromaVectorStore para búsqueda vectorial.

    Example:
        >>> store = ChromaVectorStore()
        >>> chain = RAGChain(store)
        >>> results = chain.retrieve("¿Qué es ML?")
    """

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store
        self._llm = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def retrieve(self, query: str) -> list[tuple[DocumentChunk, float]]:
        """Recupera chunks con scores de similitud desde ChromaVectorStore.

        Args:
            query: Pregunta en lenguaje natural.

        Returns:
            Lista de tuplas ``(DocumentChunk, score)`` con score en [0, 1].
        """
        raw = self._vector_store.retrieve_with_scores(query)
        return [
            (
                DocumentChunk(
                    id=str(i),
                    content=doc.page_content,
                    source=doc.metadata.get("source", ""),
                    metadata=doc.metadata,
                ),
                score,
            )
            for i, (doc, score) in enumerate(raw)
        ]

    def generate_answer(self, query: str, chunks: list[DocumentChunk]) -> str:
        """Genera una respuesta con el LLM usando el contexto dado.

        Args:
            query: Pregunta original del usuario.
            chunks: Chunks de contexto para el prompt aumentado.

        Returns:
            Respuesta generada por el LLM como string.
        """
        context = "\n\n---\n\n".join(
            f"[Fuente: {c.source}]\n{c.content}" for c in chunks
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Contexto:\n{context}\n\nPregunta: {query}"),
        ]
        response = self._llm.invoke(messages)
        return response.content
